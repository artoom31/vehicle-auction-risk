# Vehicle Auction Risk Classification

**Predicting "bad buys" at wholesale car auctions — with every model and every metric implemented from scratch and validated against scikit-learn.**

The question driving this project is not "what scores highest" but **which model still works when the data moves**. The dataset spans two years of auctions; the answer turns out to favour the simplest model in the comparison, and the last section explains why.

---

## Problem

A dealer buying at a wholesale auto auction risks acquiring a vehicle with serious undisclosed problems (a *kick*). The task is to score each transaction with the probability that it is a bad buy.

This is a **rare-event scoring problem**: a heavily imbalanced positive class (~12%), a mix of high-cardinality categorical and numeric features, and a strict temporal ordering. Structurally it is the same shape as credit-default scoring, which is why **Gini** — the industry-standard ranking metric, `2·AUC − 1` — is used throughout rather than accuracy.

## Data

Kaggle [*Don't Get Kicked!*](https://www.kaggle.com/c/DontGetKicked) (Carvana), `training.csv` — 72,983 transactions, 31 features, target `IsBadBuy`.

The dataset is not committed. See [`data/README.md`](./data/README.md) for download instructions.

| Split | Rows | Date range | Positive rate |
|---|---|---|---|
| train | 24,084 | 2009-01-05 → 2009-09-11 | 11.4% |
| valid | 24,084 | 2009-09-11 → 2010-05-11 | 13.1% |
| test | 24,815 | 2010-05-11 → 2010-12-30 | 12.3% |

## Methodology

**1. Temporal split, not random.** The data is ordered by `PurchDate` and cut into thirds. A random split would let the model learn from future auctions to predict past ones — a leak that inflates offline metrics and disappears in production.

**2. Leakage-safe encoding.** All encoders are fit on train only:
- one-hot for low-cardinality categoricals (`Auction`, `WheelType`, `Size`, `Color`, `VNST`, …)
- count encoding with an explicit fallback for categories unseen in train (`Make`, `Model`, `Trim`, `SubModel`, `BYRNO`, `VNZIP1`)

This immediately surfaced a real trap: `PRIMEUNIT` and `AUCGUART` are **99.99% missing in train but 13.2% populated in test**. A model trained on them learns nothing usable and then behaves unpredictably on the test period — both features were dropped.

**3. Feature engineering.** 13 derived features: price ratios (`VehBCost` against each of the four MMR reference prices, current vs. acquisition, clean vs. average, retail vs. auction), `odo_per_year`, `ratio_warranty_to_cost`, and group-mean deviations (`VehBCost_dev_from_Model`, `Model_mean_VehicleAge`, `Size_mean_VehBCost`, …).

**4. Model selection and tuning.** Regularization path over `C`, L1 vs L2 penalty, `k` sweep for KNN, depth sweeps for trees, early stopping for the boosting libraries — all selected on the validation third, with the test third touched once.

## What is implemented from scratch

Everything below is written in NumPy and numerically compared against the scikit-learn equivalent.

| Component | Notes |
|---|---|
| `MyLogisticRegression` | mini-batch SGD, L2 penalty, hand-derived NLL gradient `(p − y)` |
| `MyGaussianNB` | variance smoothing, joint log-likelihood |
| `MyKNNClassifier` | batched distance computation |
| `Node` | Gini impurity (classification) / std reduction (regression), best-split search |
| `DecisionTreeClassifier` / `DecisionTreeRegressor` | CART with `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` |
| `RandomForestClassifier` | bagging over the above, seeded and reproducible |
| `ExtraTreesClassifier` | randomized split thresholds |
| `GradientBoostingClassifier` | binary cross-entropy gradient, `decision_function` → sigmoid → `predict_proba` |
| Metrics | ROC-AUC, Gini, Precision, Recall, F1, AUC-PR |

**Metric verification** — all four metrics reproduce scikit-learn exactly:

| | my implementation | sklearn | abs diff |
|---|---|---|---|
| Precision @ 0.5 | 0.631961 | 0.631961 | 0.0 |
| Recall @ 0.5 | 0.244458 | 0.244458 | 0.0 |
| F1 @ 0.5 | 0.352544 | 0.352544 | 0.0 |
| AUC-PR | 0.401153 | 0.401153 | 0.0 |

Gini likewise matches `2·roc_auc_score − 1` to 0.0 across all three baseline models.

## Results

**Own decision tree vs. scikit-learn** (valid Gini):

| max_depth | mine | sklearn |
|---|---|---|
| 3 | 0.4284 | 0.4347 |
| 5 | 0.4352 | 0.4398 |
| 7 | 0.4256 | 0.4295 |
| 9 | 0.4031 | 0.4024 |

**Own ensembles vs. production GBDT libraries** (valid Gini):

| Model | valid Gini |
|---|---|
| own decision tree (depth 7) | 0.4256 |
| own Random Forest (100 trees, depth 12) | 0.4622 |
| **own Gradient Boosting (200 trees, depth 4)** | **0.4833** |
| LightGBM (best_iter 152) | 0.4927 |
| XGBoost (best_iter 194) | 0.4931 |
| XGBoost DART | 0.4900 |
| CatBoost (best_iter 568) | 0.4956 |

The hand-written boosting implementation lands **0.012 Gini** behind tuned CatBoost — close enough to confirm the gradient derivation is correct, and the gap is a fair measure of what histogram binning, leaf-wise growth and native categorical handling actually buy you.

**Final models on the untouched test third:**

| Model | train Gini | valid Gini | test Gini | test AUC-PR |
|---|---|---|---|---|
| L1 logistic regression (`C = 0.03`) + engineered features | 0.5181 | 0.4796 | **0.4926** | 0.4487 |
| CatBoost | 0.5968 | 0.4956 | 0.4757 | — |
| Gaussian NB | — | 0.3842 | 0.3438 | 0.1993 |
| KNN (k = 400) | — | 0.4524 | 0.4653 | 0.4172 |

## Conclusion

**The regularized linear model generalizes better than CatBoost on the held-out period despite scoring lower on validation.**

| | train → test Gini gap |
|---|---|
| L1 logistic regression | 0.5181 → 0.4926 = **0.026** |
| CatBoost | 0.5968 → 0.4757 = **0.121** |

CatBoost wins validation by 0.016 Gini and loses the test period by 0.017. Validation sits between train and test in time, so it is partly protected by proximity to the training window; the test third is a full year later, and that is where the extra capacity turns into memorized structure that no longer holds.

The practical reading: on a dataset that drifts, **selecting on validation score alone picks the wrong model**. The train-to-test gap is the more honest selection criterion, and it is the reason the whole project is built on a chronological split — a random split would have hidden the effect completely and handed the decision to CatBoost.

## Layout

```
src/
  classification.ipynb    # ML4 — split, encoding, from-scratch models & metrics, tuning
  ML5_decision_trees.ipynb # ML5 — trees, ensembles, GBDT library comparison
  trees.py                # from-scratch tree & ensemble library (~400 lines)
data/
  README.md               # how to obtain training.csv
```

## How to run

```bash
pip install numpy pandas scikit-learn lightgbm xgboost catboost matplotlib jupyter
```

Place `training.csv` in `data/`, then run `src/classification.ipynb` followed by `src/ML5_decision_trees.ipynb`. All random seeds are fixed; results are reproducible.

## Notes

Built during the School 21 Machine Learning specialization and defended in live peer review — every decision above had to be justified orally to reviewers who had the code open in front of them. Notebook commentary is in Russian.
