"""ML5. Decision trees and ensembles, implemented from scratch on NumPy.

Classes:
    Node                       -- tree node: holds data, computes impurity, finds splits
    DecisionTreeClassifier     -- CART classifier (Gini impurity)
    DecisionTreeRegressor      -- CART regressor (standard deviation reduction)
    RandomForestClassifier     -- bagging over decision trees
    ExtraTreesClassifier       -- random forest with random-split trees
    GradientBoostingClassifier -- GBDT for binary classification (BCE loss)
"""

import numpy as np

EPS = 1e-12


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


class Node:
  

    __slots__ = ("X", "y", "depth", "left", "right", "feature", "threshold", "value")

    def __init__(self, X, y, depth=0):
        self.X = X
        self.y = y
        self.depth = depth
        self.left = None
        self.right = None
        self.feature = None    # index of the split feature
        self.threshold = None  # go left if x[feature] <= threshold
        self.value = None      # leaf prediction (class probabilities or mean)

    @property
    def is_leaf(self):
        return self.left is None

    
    def gini(self):
        
        counts = np.bincount(self.y.astype(np.int64))
        p = counts / len(self.y)
        return float(1.0 - np.sum(p * p))

    def std(self):
        """Standard deviation of the node targets (regression impurity)."""
        return float(np.std(self.y))

    def impurity(self, criterion):
        return self.gini() if criterion == "gini" else self.std()

    # ------------------------------------------------------------- best split
    def find_best_split(self, feature_ids, criterion="gini",
                        min_samples_leaf=1, n_classes=None):
   
        X, y = self.X, self.y
        n = len(y)
        if n < 2 * min_samples_leaf:
            return None
        parent = self.impurity(criterion)
        sizes_l = np.arange(1, n)
        sizes_r = n - sizes_l
        size_ok = (sizes_l >= min_samples_leaf) & (sizes_r >= min_samples_leaf)

        best_gain, best_feature, best_thr = EPS, None, None
        for f in feature_ids:
            x = X[:, f]
            order = np.argsort(x)
            xs = x[order]
            if xs[0] == xs[-1]:  # constant feature
                continue
            ys = y[order]
            if criterion == "gini":
                child = self._children_gini(ys, sizes_l, sizes_r, n_classes)
            else:
                child = self._children_std(ys, sizes_l, sizes_r)
            gains = parent - child / n
            valid = size_ok & (xs[1:] != xs[:-1])
            gains[~valid] = -np.inf
            j = int(np.argmax(gains))
            if gains[j] > best_gain:
                best_gain = float(gains[j])
                best_feature = f
                best_thr = float((xs[j] + xs[j + 1]) / 2.0)
        if best_feature is None:
            return None
        return best_feature, best_thr, best_gain

    def find_random_split(self, feature_ids, rng, criterion="gini",
                          min_samples_leaf=1, n_classes=None):
        """Extra-randomized split search: for every candidate feature a single
        random threshold is drawn uniformly between min and max, the best
        candidate is returned. Returns ``(feature, threshold, gain)`` or ``None``.
        """
        X, y = self.X, self.y
        n = len(y)
        if n < 2 * min_samples_leaf:
            return None
        parent = self.impurity(criterion)

        best_gain, best_feature, best_thr = EPS, None, None
        for f in feature_ids:
            x = X[:, f]
            lo, hi = x.min(), x.max()
            if lo == hi:
                continue
            thr = rng.uniform(lo, hi)
            mask = x <= thr
            nl = int(mask.sum())
            nr = n - nl
            if nl < min_samples_leaf or nr < min_samples_leaf:
                continue
            yl, yr = y[mask], y[~mask]
            if criterion == "gini":
                cl = np.bincount(yl.astype(np.int64), minlength=n_classes)
                cr = np.bincount(yr.astype(np.int64), minlength=n_classes)
                il = 1.0 - np.sum((cl / nl) ** 2)
                ir = 1.0 - np.sum((cr / nr) ** 2)
            else:
                il, ir = np.std(yl), np.std(yr)
            gain = parent - (nl * il + nr * ir) / n
            if gain > best_gain:
                best_gain, best_feature, best_thr = float(gain), f, float(thr)
        if best_feature is None:
            return None
        return best_feature, best_thr, best_gain

    # ------------------------------------------------------ vectorized scans
    @staticmethod
    def _children_gini(ys, sizes_l, sizes_r, n_classes):
        """n * weighted Gini of children for every split position."""
        gl = np.ones(len(sizes_l))
        gr = np.ones(len(sizes_r))
        for k in range(n_classes):
            cum = np.cumsum(ys == k)
            left_k = cum[:-1]
            right_k = cum[-1] - left_k
            gl -= (left_k / sizes_l) ** 2
            gr -= (right_k / sizes_r) ** 2
        return sizes_l * gl + sizes_r * gr

    @staticmethod
    def _children_std(ys, sizes_l, sizes_r):
        """n * weighted std of children for every split position."""
        s = np.cumsum(ys)
        s2 = np.cumsum(ys * ys)
        sum_l, sum2_l = s[:-1], s2[:-1]
        sum_r, sum2_r = s[-1] - sum_l, s2[-1] - sum2_l
        var_l = np.maximum(sum2_l / sizes_l - (sum_l / sizes_l) ** 2, 0.0)
        var_r = np.maximum(sum2_r / sizes_r - (sum_r / sizes_r) ** 2, 0.0)
        return sizes_l * np.sqrt(var_l) + sizes_r * np.sqrt(var_r)


class BaseDecisionTree:
    """Common CART logic: greedy recursive growing of a binary tree."""

    _criterion = None  # "gini" or "std", set by subclasses

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, splitter="best", random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features  # None | int | float | "sqrt" | "log2"
        self.splitter = splitter          # "best" | "random"
        self.random_state = random_state

    # ------------------------------------------------------------------- fit
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = self._prepare_target(np.asarray(y))
        self.n_features_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self._k_features = self._resolve_max_features()

        self.root_ = Node(X, y, depth=0)
        stack = [self.root_]
        while stack:
            node = stack.pop()
            node.value = self._leaf_value(node.y)
            if not self._should_stop(node):
                split = self._find_split(node)
                if split is not None:
                    node.feature, node.threshold, _ = split
                    mask = node.X[:, node.feature] <= node.threshold
                    node.left = Node(node.X[mask], node.y[mask], node.depth + 1)
                    node.right = Node(node.X[~mask], node.y[~mask], node.depth + 1)
                    stack.append(node.left)
                    stack.append(node.right)
            node.X = node.y = None  # free memory, data lives in the children
        return self

    def _find_split(self, node):
        if self._k_features < self.n_features_:
            feats = self._rng.choice(self.n_features_, size=self._k_features,
                                     replace=False)
        else:
            feats = range(self.n_features_)
        kwargs = dict(criterion=self._criterion,
                      min_samples_leaf=self.min_samples_leaf,
                      n_classes=getattr(self, "n_classes_", None))
        if self.splitter == "random":
            return node.find_random_split(feats, rng=self._rng, **kwargs)
        return node.find_best_split(feats, **kwargs)

    def _should_stop(self, node):
        if self.max_depth is not None and node.depth >= self.max_depth:
            return True
        if len(node.y) < self.min_samples_split:
            return True
        return node.impurity(self._criterion) < EPS  # pure node

    def _resolve_max_features(self):
        mf, n = self.max_features, self.n_features_
        if mf is None:
            return n
        if mf == "sqrt":
            return max(1, int(np.sqrt(n)))
        if mf == "log2":
            return max(1, int(np.log2(n)))
        if isinstance(mf, float):
            return max(1, int(mf * n))
        return min(int(mf), n)

    # --------------------------------------------------------------- predict
    def _predict_values(self, X, value_dim):
        X = np.asarray(X, dtype=np.float64)
        out = np.empty((len(X), value_dim))
        self._route(self.root_, X, np.arange(len(X)), out)
        return out

    def _route(self, node, X, idx, out):
        if len(idx) == 0:
            return
        if node.is_leaf:
            out[idx] = node.value
            return
        mask = X[idx, node.feature] <= node.threshold
        self._route(node.left, X, idx[mask], out)
        self._route(node.right, X, idx[~mask], out)


class DecisionTreeClassifier(BaseDecisionTree):
    """CART classifier with the Gini impurity criterion.

    >>> model = DecisionTreeClassifier(max_depth=7)
    >>> model.fit(Xtrain, ytrain)
    >>> model.predict_proba(Xvalid)
    """

    _criterion = "gini"

    def _prepare_target(self, y):
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return y_enc.astype(np.int64)

    def _leaf_value(self, y):
        return np.bincount(y, minlength=self.n_classes_) / len(y)

    def predict_proba(self, X):
        return self._predict_values(X, self.n_classes_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class DecisionTreeRegressor(BaseDecisionTree):
    """CART regressor (MSE loss): splits by standard deviation reduction,
    leaves predict the mean target."""

    _criterion = "std"

    def _prepare_target(self, y):
        return np.asarray(y, dtype=np.float64)

    def _leaf_value(self, y):
        return np.array([np.mean(y)])

    def predict(self, X):
        return self._predict_values(X, 1).ravel()


class RandomForestClassifier:

    def __init__(self, n_trees=100, max_depth=None, max_features="sqrt",
                 subsample=0.66, min_samples_split=2, min_samples_leaf=1,
                 splitter="best", random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.splitter = splitter
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        k = min(n, max(1, int(round(self.subsample * n))))
        self.trees_ = []
        for _ in range(self.n_trees):
            idx = rng.choice(n, size=k, replace=False) if k < n else np.arange(n)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                splitter=self.splitter,
                random_state=int(rng.integers(2 ** 31)),
            )
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        proba = np.zeros((len(X), len(self.classes_)))
        for tree in self.trees_:
            cols = np.searchsorted(self.classes_, tree.classes_)
            proba[:, cols] += tree.predict_proba(X)
        return proba / len(self.trees_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class ExtraTreesClassifier(RandomForestClassifier):
    """Extremely randomized trees: no row subsampling, every tree uses random
    thresholds instead of the exhaustive best-split search."""

    def __init__(self, n_trees=100, max_depth=None, max_features="sqrt",
                 min_samples_split=2, min_samples_leaf=1, random_state=None):
        super().__init__(n_trees=n_trees, max_depth=max_depth,
                         max_features=max_features, subsample=1.0,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         splitter="random", random_state=random_state)


class GradientBoostingClassifier:


    def __init__(self, number_of_trees=100, learning_rate=0.1, max_depth=3,
                 max_features=None, subsample=1.0, min_samples_leaf=1,
                 random_state=None):
        self.number_of_trees = number_of_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features = max_features
        self.subsample = subsample
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)  # binary target: 0 / 1
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        k = min(n, max(1, int(round(self.subsample * n))))

        p0 = np.clip(y.mean(), EPS, 1.0 - EPS)
        self.f0_ = float(np.log(p0 / (1.0 - p0)))  # prior log-odds
        F = np.full(n, self.f0_)

        self.trees_ = []
        for _ in range(self.number_of_trees):
            antigrad = y - _sigmoid(F)  # -dL/dF for the BCE loss
            idx = rng.choice(n, size=k, replace=False) if k < n else np.arange(n)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=int(rng.integers(2 ** 31)),
            )
            tree.fit(X[idx], antigrad[idx])
            F += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        F = np.full(len(X), self.f0_)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X):
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)
