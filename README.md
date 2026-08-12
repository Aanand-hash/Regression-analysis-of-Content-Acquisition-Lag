# 📺 Predicting Time-to-Netflix: A Regression Analysis of Content Acquisition Lag

Predicting how many years after its original release a movie or TV show gets added to Netflix, using metadata alone.

## Project Goal

When a title is released, it doesn't immediately show up on Netflix — some titles appear within months, others take years. This project builds a regression model to predict **Content Acquisition Lag**: the number of years between a title's original release and the date it was added to Netflix.

Rather than optimizing purely for accuracy, this project emphasizes the full ML workflow — data understanding, cleaning, feature engineering, model interpretation, and honestly stating limitations.

## Problem Statement

> Can we estimate how long it will take for a movie or TV show to appear on Netflix using only its metadata?

- **Target variable**: `content_acquisition_lag`
- **Formula**: `Acquisition Lag = Year(Date Added) − Release Year`
- **Type**: Supervised regression (continuous target)

## Dataset

- **Source**: Netflix Movies and TV Shows Dataset (Snapshot: September 2021)
- ~8,800 titles with metadata: director, cast, country, release year, date added, content rating, duration, genres, description
- Does **not** contain: user ratings, popularity, watch time, or revenue

## Assumptions

- `release_year` represents the title's first public release
- The year extracted from `date_added` represents the acquisition year
- Missing metadata is handled during preprocessing
- Predictions rely solely on publicly available metadata (no licensing/business data)

## Workflow

1. **EDA** — distribution of release years, rating counts, missing values, duplicates
2. **Data cleaning** — dropped duplicates, filled missing `director`/`cast`/`country` with `"Unknown"`, dropped rows with missing `date_added`/`duration`, filled missing `rating` with the mode
3. **Feature engineering** — parsed `date_added` into `added_year` and `added_month`; computed the target `content_acquisition_lag`
4. **Feature selection** — kept `release_year`, `added_month`, `type`, `rating` as inputs
5. **Encoding** — one-hot encoded `type` and `rating` with `pd.get_dummies(drop_first=True)`, no feature scaling applied
6. **Modeling** — `sklearn.linear_model.LinearRegression()`, default params, 80/20 train/test split (`random_state=42`)
7. **Evaluation** — MAE, RMSE, R² on the test set, plus an Actual vs Predicted scatter plot

## Model Details

| Item | Value |
|---|---|
| Model | `sklearn.linear_model.LinearRegression()` (default params) |
| Features | 16 (2 numeric + 14 one-hot from `type`/`rating`) |
| Scaling | None — raw ints and 0/1 dummy columns |
| Train/Test split | 80/20, `random_state=42` |

## Results

| Metric | Value |
|---|---|
| MAE | 1.24 years |
| RMSE | 1.52 years |
| R² (test) | 0.9769 |
| R² (train) | 0.9689 |

Train and test R² are close, suggesting the model generalizes well without significant overfitting. Most predictions fall close to the ideal diagonal in the Actual vs Predicted plot.

## Limitations

Real-world factors such as licensing agreements, regional availability, and Netflix's internal business decisions aren't captured in this metadata and likely explain the remaining prediction error.

## Repository Structure

```
.
├── a-regression-analysis-of-content-acquisition-lag.ipynb   # Full Python/ML pipeline
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```



## Author's Note

This notebook was as much about building ML intuition as it was about the model itself — understanding the data, engineering features, applying one-hot encoding correctly, and interpreting results critically, rather than treating each step as an isolated exercise.
