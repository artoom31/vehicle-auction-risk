# Скоринг риска на автоаукционе

**Предсказание неудачных покупок на оптовом автоаукционе. Все модели и метрики написаны с нуля на NumPy и сверены с scikit-learn.**

Главный вопрос проекта — не «что даёт лучшую метрику», а **какая модель продолжает работать, когда данные смещаются во времени**.

## Задача

Дилер на оптовом аукционе рискует купить машину со скрытыми дефектами. Нужно оценить вероятность такой покупки.

Это скоринг редкого события: положительный класс ~12%, категориальные признаки высокой мощности, строгий порядок во времени. По структуре — то же самое, что скоринг дефолта по кредиту, поэтому основная метрика здесь **Gini** (`2·AUC − 1`), а не accuracy.

Данные: Kaggle [*Don't Get Kicked!*](https://www.kaggle.com/c/DontGetKicked) — 72 983 сделки, 31 признак. В репозиторий не коммитятся, инструкция в [`data/README.md`](./data/README.md).

## Методология

**Разбиение по времени, а не случайное.** Данные упорядочены по `PurchDate` и разрезаны на трети. Случайное разбиение позволило бы модели учиться на будущих аукционах — утечка, которая завышает офлайн-метрики и исчезает в продакшене.

**Кодировщики обучаются только на train.** Это сразу вскрыло ловушку: `PRIMEUNIT` и `AUCGUART` заполнены на 0.01% в train и на 13.2% в test. Модель на них ничему не учится, а на тесте ведёт себя непредсказуемо — оба признака выброшены.

**13 сконструированных признаков:** отношения цены к четырём справочным ценам MMR, пробег на год, отклонения от групповых средних.

## Реализовано с нуля

Логистическая регрессия (mini-batch SGD, L2, градиент NLL выведен руками), гауссовский наивный байес, батчевый KNN, CART-классификатор и регрессор, случайный лес, Extra Trees, градиентный бустинг.

Метрики ROC-AUC, Gini, precision, recall, F1 и AUC-PR воспроизводят scikit-learn с расхождением **0.0000**.

## Результаты

Свои ансамбли против промышленных библиотек, valid Gini:

| Модель | valid Gini |
|---|---|
| своё дерево (глубина 7) | 0.4256 |
| свой случайный лес | 0.4622 |
| **свой градиентный бустинг** | **0.4833** |
| LightGBM | 0.4927 |
| XGBoost | 0.4931 |
| CatBoost | 0.4956 |

Отставание своего бустинга от настроенного CatBoost — 0.012 Gini. Этого достаточно, чтобы подтвердить корректность вывода градиента, а сам разрыв показывает, что дают гистограммное биннинг, leaf-wise рост и нативная работа с категориями.

## Главный вывод

**Регуляризованная линейная модель обобщается лучше CatBoost на отложенном периоде, хотя проигрывает ему на валидации.**

| Модель | train → test Gini | разрыв |
|---|---|---|
| L1-логрегрессия | 0.5181 → 0.4926 | **0.026** |
| CatBoost | 0.5968 → 0.4757 | **0.121** |

CatBoost выигрывает валидацию на 0.016 Gini и проигрывает тест на 0.017. Валидация лежит во времени между train и test и потому частично защищена близостью к обучающему окну. Тестовая треть — на год позже, и там лишняя ёмкость превращается в запомненную структуру, которой больше нет.

Практический смысл: на дрейфующих данных **выбор модели по одной только валидации даёт неверный ответ**. Разрыв train→test честнее. Именно поэтому весь проект построен на хронологическом разбиении — случайное скрыло бы эффект полностью.

## Структура и запуск

```
src/classification.ipynb     # разбиение, кодирование, свои модели и метрики
src/ML5_decision_trees.ipynb # деревья, ансамбли, сравнение с GBDT-библиотеками
src/trees.py                 # своя библиотека деревьев и ансамблей (~400 строк)
```

```bash
pip install numpy pandas scikit-learn lightgbm xgboost catboost matplotlib jupyter
```

Положите `training.csv` в `data/` и запустите ноутбуки в порядке выше. Все seed зафиксированы, результаты воспроизводимы.

Проект сделан в рамках специализации Machine Learning Школы 21 и защищён на устном peer review. Комментарии в ноутбуках — на русском.

---

# English

**Predicting "bad buys" at wholesale car auctions. Every model and every metric implemented from scratch in NumPy and validated against scikit-learn.**

The driving question is not "what scores highest" but **which model still works when the data moves**.

## Problem

A dealer at a wholesale auction risks buying a vehicle with undisclosed defects. The task is to score the probability of that.

This is rare-event scoring: ~12% positive class, high-cardinality categoricals, strict temporal ordering. Structurally identical to credit-default scoring, which is why **Gini** (`2·AUC − 1`) is the metric throughout rather than accuracy.

Data: Kaggle [*Don't Get Kicked!*](https://www.kaggle.com/c/DontGetKicked) — 72,983 transactions, 31 features. Not committed; see [`data/README.md`](./data/README.md).

## Methodology

**Chronological split, not random.** Ordered by `PurchDate` and cut into thirds. A random split would let the model learn from future auctions — a leak that inflates offline metrics and vanishes in production.

**Encoders fit on train only.** This surfaced a real trap: `PRIMEUNIT` and `AUCGUART` are 0.01% populated in train and 13.2% in test. Both dropped.

**13 engineered features:** price ratios against the four MMR reference prices, odometer per year, deviations from group means.

## Implemented from scratch

Logistic regression (mini-batch SGD, L2, hand-derived NLL gradient), Gaussian Naive Bayes, batched KNN, CART classifier and regressor, Random Forest, Extra Trees, gradient boosting.

ROC-AUC, Gini, precision, recall, F1 and AUC-PR all reproduce scikit-learn to **0.0000**.

## Results

Own ensembles against production libraries, valid Gini:

| Model | valid Gini |
|---|---|
| own tree (depth 7) | 0.4256 |
| own Random Forest | 0.4622 |
| **own Gradient Boosting** | **0.4833** |
| LightGBM | 0.4927 |
| XGBoost | 0.4931 |
| CatBoost | 0.4956 |

The hand-written booster lands 0.012 Gini behind tuned CatBoost — close enough to confirm the gradient derivation, and the gap measures what histogram binning, leaf-wise growth and native categorical handling actually buy.

## Conclusion

**The regularized linear model generalizes better than CatBoost on the held-out period despite scoring lower on validation.**

| Model | train → test Gini | gap |
|---|---|---|
| L1 logistic regression | 0.5181 → 0.4926 | **0.026** |
| CatBoost | 0.5968 → 0.4757 | **0.121** |

CatBoost wins validation by 0.016 Gini and loses the test period by 0.017. Validation sits between train and test in time, so it is partly protected by proximity to the training window. The test third is a full year later, and that is where extra capacity turns into memorized structure.

The practical reading: on drifting data, **selecting on validation score alone picks the wrong model**. The train-to-test gap is the more honest criterion — and the reason the project is built on a chronological split, which a random split would have hidden entirely.

## Layout and running

```
src/classification.ipynb      # split, encoding, from-scratch models and metrics
src/ML5_decision_trees.ipynb  # trees, ensembles, GBDT library comparison
src/trees.py                  # from-scratch tree and ensemble library (~400 lines)
```

```bash
pip install numpy pandas scikit-learn lightgbm xgboost catboost matplotlib jupyter
```

Place `training.csv` in `data/` and run the notebooks in that order. All seeds fixed; results reproducible.

Built during the School 21 Machine Learning specialization and defended in live peer review. Notebook commentary is in Russian.
