# Comparing Classifiers: Bank Telemarketing

** Practical Application III - UC Berkely Professional Certificate in Machine Learning/AI **
Notebok: prompt_III.ipynb
Completed Script: Compare_classifiers.ipynb
Dataset:[data/bank-additional-full.csv](./data/bank-additional-full.csv)([UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing))
Background:['CRISP-DM-BANK.pdf](./CRISP-DM-BANK.pdf) . attribute notes in [data/bank-additional-namex.txt]

## Overview
Portugese bank telemarketing contacts (May-2008-November 2010) are used to predict whether a client will subscribe
to a **term deposit** ('y' = yes/no). The assignment walks through CRISP-DM style steps-understanding data, encoding features, baseline and logistic regression models, then comparing
**logistic regression**, **KNN**, **decision trees**, and **SVM**, with optional tuning and a better metric for imbalanced classes

## Business objective

Help the bank **prioritize outbound calls** and reduce wasted agent time by estimating subscription likelihood **before** a call, using client and campaign history—not call duration (which is only known after the call and leaks the outcome).

## Data

| Item | Detail |
|------|--------|
| Rows | 41,188 (`bank-additional-full.csv`) |
| Target | ~88.7% `no`, ~11.3% `yes` (imbalanced) |
| Missing values | No literal nulls; several categoricals use label `unknown` |
| Modeling features | Bank-client + campaign fields only in `compare_classifiers.py` (no macro indicators; **`duration` excluded** for a realistic pre-call model) |

Smaller sample for quick experiments: [`data/bank-additional.csv`](./data/bank-additional.csv) (~10% stratified sample).

## What we did

1. **EDA** — Campaign timeframe, dtypes, and `unknown` counts by column.  
2. **Feature engineering** — Scale numerics (`age`, `campaign`, `pdays`, `previous`); one-hot encode categoricals.  
3. **Split** — 80/20 stratified train/test (`random_state=42`).  
4. **Baseline** — `DummyClassifier` (majority class) ~88.7% test accuracy.  
5. **Logistic regression** — Simple linear benchmark (~89% test accuracy).  
6. **Model comparison** — Fit time, train and test accuracy for LR, KNN, default decision tree, and SVM (SVM trained on a stratified subsample for runtime on large `n`).  
7. **Improvement** — `GridSearchCV` on a decision tree with `class_weight='balanced'` and **`scoring='f1'`** to better detect subscribers (minority class).

## Key takeaway

High **accuracy** is misleading here: a model that always predicts “no” is already ~89% accurate. For marketing, **recall/F1 on “yes”** matters more—catching potential subscribers without calling everyone. Tuned tree trades some accuracy for stronger detection of positives vs. default logistic regression on F1.

## Repository contents

```
Module17/
├── README.md
├── prompt_III.ipynb          <- course starter (Problems 1–11)
├── compare_classifiers.py      <- optional: executes notebook via Jupyter
├── CRISP-DM-BANK.pdf
├── notes
└── data/
    ├── bank-additional-full.csv
    ├── bank-additional.csv
    └── bank-additional-names.txt
```

## Run locally

Open **`prompt_III.ipynb`** in Jupyter / VS Code and **Run All** (first run may take several minutes because of SVM).

Requirements: Python 3, `pandas`, `scikit-learn`.

## Tools used

Python, pandas, scikit-learn (`ColumnTransformer`, `OneHotEncoder`, `StandardScaler`, `GridSearchCV`, classifiers and classification metrics).

## Citation

If you use this dataset in write-ups, cite Moro, Cortez, and Rita (2014), *Decision Support Systems* — see [`data/bank-additional-names.txt`](./data/bank-additional-names.txt).