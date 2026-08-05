# Kepler Exoplanet Classification

A machine learning project to classify Kepler Objects of Interest (KOIs) into one of three dispositions — **CANDIDATE**, **CONFIRMED**, or **FALSE POSITIVE** — based on the photometric and stellar parameters recorded by NASA's Kepler mission.

## Motivation

Every signal Kepler picks up isn't necessarily a planet — a lot of them turn out to be eclipsing binaries, instrumental noise, or other artifacts. NASA's own pipeline assigns each detected signal a disposition, but I wanted to see how well a model could learn to make that same call using only the underlying measurements — orbital period, transit depth, stellar parameters, and so on — without ever touching the dataset's own precomputed disposition score. That last part matters: `koi_score` and `koi_pdisposition` are outputs of NASA's own vetting model, so using them as inputs would just be leaking the answer straight into the features.

## Dataset

The raw dataset has 50 columns and a mix of orbital, transit, and stellar measurements, along with their associated uncertainty (error) columns. Two columns were entirely empty and dropped immediately. A few identifier/name columns (`kepoi_name`, `kepler_name`, row index) were also dropped early — they're unique per row and carry no real predictive signal, and one-hot encoding something like `kepoi_name` would have blown up the feature space for nothing.

## Approach

### 1. Exploratory analysis
Started with the basics — shape, dtypes, missing value percentages, duplicate check. No duplicates, but a good chunk of missing data across several columns, which meant imputation was unavoidable.

### 2. Feature relevance — statistical testing
Rather than dropping or keeping columns on gut feeling, I ran the actual hypothesis tests:
- **ANOVA (`f_oneway`)** for numerical features against the three-class target, since I've got more than two categories to compare
- **Chi-squared test** for the one remaining categorical feature (`koi_tce_delivname`) against the target

Everything with p < 0.05 was kept. This left 37 numerical columns confirmed as statistically associated with the disposition, and the chi-squared test confirmed the categorical column was relevant too.

### 3. Train/test split — done early, on purpose
The split happens *before* any imputation or outlier handling. This is deliberate — if you impute or cap outliers using statistics computed across the whole dataset, information from the test set leaks into training, and your evaluation numbers stop meaning anything.

### 4. Missing value imputation
- **Numerical columns:** KNN imputer, wrapped inside a `Pipeline`/`ColumnTransformer` (with scaling applied first, since KNN imputation is distance-based and needs features on a comparable scale). Fitting this inside the pipeline means it's refit on each training fold during cross-validation, not just once on the full training set.
- **Categorical column:** random sample imputation instead of KNN, so the original distribution of categories is preserved rather than skewed toward a single most-common value.

### 5. Outlier handling
Looked at distribution and box plots per column and capped a handful of the more skewed ones (`koi_period`, `koi_depth`, `koi_prad`, `koi_teq`, `koi_insol`, `koi_slogg`, `koi_srad`) at their 99th percentile — computed **from the training set only**, then applied to both train and test using that same fixed threshold. Error columns were deliberately left untouched, since measurement error naturally varies with instrument precision for objects this far away, and squashing that signal would throw away real information.

### 6. Feature engineering
Rather than feeding the model raw `err1`/`err2` pairs as if they were unrelated numbers, I combined each pair into:
- an **absolute uncertainty** feature (`err1 - err2`)
- a **relative uncertainty** feature (that difference normalized by the corresponding measurement)

This gives the model a more directly meaningful signal — "how uncertain is this measurement" — instead of leaving it to infer that relationship from two disconnected error columns on its own.

### 7. Modeling
Compared five classifiers, each tuned via `GridSearchCV` (5-fold, scored on macro F1 to account for class imbalance):

| Model | Notes |
|---|---|
| Logistic Regression | tuned over `C` |
| Random Forest | tuned over `n_estimators`, `max_depth` |
| Gradient Boosting | tuned over `n_estimators`, `learning_rate` |
| SVC | |
| KNN | |

Every model shares the same pipeline structure: preprocessing (impute + scale) → `SelectKBest` (k=37, matching the ANOVA-confirmed feature count) → classifier. Keeping this inside one `Pipeline` object means feature selection and imputation are refit per fold, so there's no leakage from the test set into model selection either.

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| **Gradient Boosting** | **89.96%** | **0.867** |
| Random Forest | 88.97% | 0.853 |
| SVC | 88.87% | 0.851 |
| Logistic Regression | 88.55% | 0.848 |
| KNN | 84.16% | 0.793 |

Gradient Boosting came out on top, though the gap over Random Forest and SVC isn't huge. KNN lagged the rest, which suggests the class boundaries here aren't purely a matter of proximity in feature space.

The more interesting pattern shows up when you look at per-class performance instead of the overall numbers. Every single model separates `FALSE POSITIVE` almost perfectly (precision/recall around 0.96–0.98), but all of them struggle with `CANDIDATE` vs `CONFIRMED` — `CANDIDATE` recall drops as low as 0.62–0.78 depending on the model. That's not really a modeling failure — it reflects how these labels actually get assigned in practice. A `FALSE POSITIVE` is usually a clear-cut artifact (an eclipsing binary, instrumental noise), while a `CANDIDATE` is often just a `CONFIRMED` planet that hasn't accumulated enough follow-up observation yet. The two classes sit on a continuum rather than a hard boundary, so no amount of tuning fully resolves it — the ceiling is coming from the labels themselves, not the model.



## Tech stack

`pandas`, `numpy`, `scikit-learn`, `scipy` (for the statistical tests), `seaborn`/`matplotlib` for EDA.
