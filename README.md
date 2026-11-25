# 🛡️ UPI Fraud Detection

A machine-learning project that tries to detect fraudulent UPI (Unified Payments Interface)
transactions, with a multi-page Streamlit web app on top of the trained model.

> **Read this first: an honest result**
> The model trained on this dataset is **no better than random guessing**
> (test ROC-AUC ≈ 0.49, PR-AUC ≈ 0.0018, where random guessing gives 0.0019).
> This is **not a bug**. The analysis shows that the dataset's features carry almost no
> information about fraud (see [Why the model is weak](#9-why-the-model-is-weak)).
> The value of this project is the complete, correct **workflow**: cleaning, leak-free
> preprocessing, imbalance handling, model comparison, honest evaluation, and an app.
> Do **not** use the model for real fraud decisions.

---

## Table of contents

1. [Project at a glance](#1-project-at-a-glance)
2. [Quick start](#2-quick-start)
3. [The dataset](#3-the-dataset)
4. [Workflow](#4-workflow)
5. [Folder structure: what every file does](#5-folder-structure-what-every-file-does)
6. [Step by step: what we did and why](#6-step-by-step-what-we-did-and-why)
7. [The algorithms and why we chose them](#7-the-algorithms-and-why-we-chose-them)
8. [The graphs explained](#8-the-graphs-explained)
9. [Why the model is weak](#9-why-the-model-is-weak)
10. [The Streamlit app](#10-the-streamlit-app)
11. [Tests](#11-tests)
12. [Deployment with Docker](#12-deployment-with-docker)
13. [Limitations and future work](#13-limitations-and-future-work)
14. [Team and Git workflow](#14-team-and-git-workflow)
15. [Glossary](#15-glossary)
16. [FAQ](#16-faq)

---

## 1. Project at a glance

| Item | Value |
|---|---|
| Goal | Predict whether a UPI transaction is fraud (`fraud_flag` = 1) |
| Data | 250,000 transactions from 2024 (`data/raw/upi_transactions_2024_raw.csv`) |
| Fraud cases | 480 (0.19%), so the problem is **extremely imbalanced** |
| Problem type | Binary classification (fraud / not fraud) |
| Language / libraries | Python, pandas, scikit-learn, XGBoost, imbalanced-learn, Streamlit |
| Final model | Logistic Regression with class weights (inside a full pipeline) |
| Test result | ROC-AUC 0.488, PR-AUC 0.0018, recall 0.40, precision 0.0019 |

---

## 2. Quick start

```bash
# 1. Install the libraries
pip install -r requirements.txt

# 2. Run the pipeline (each step creates files for the next step)
python ml/src/data_cleaning.py      # data/processed/cleaned_data.csv
python ml/src/preprocessing.py      # X_train, X_test, y_train, y_test
python ml/src/train_model.py        # ml/artifacts/best_model.pkl
python ml/src/evaluate_model.py     # reports/figures/20_*.png, 21_*.png

# 3. Run the tests
python -m pytest tests

# 4. Start the app
python -m streamlit run app/streamlit_app.py
```

The notebooks in `ml/notebooks/` can be run in order (01 → 08) with Jupyter.

---

## 3. The dataset

One row = one UPI transaction. There are **17 columns**:

| Column | Type | Meaning |
|---|---|---|
| `transaction id` | ID | Unique id. Not useful for a model, so it is dropped before training |
| `timestamp` | Date/time | When the payment happened (2024-01-01 to 2024-12-30) |
| `transaction type` | Category | P2P, P2M, Bill Payment, Recharge |
| `merchant_category` | Category | Food, Fuel, Grocery, Education, ... (10 values) |
| `amount (INR)` | Number | Payment amount in rupees (Rs 10 to Rs 42,099; median Rs 629) |
| `transaction_status` | Category | SUCCESS (95%) or FAILED (5%) |
| `sender_age_group` / `receiver_age_group` | Category | 18-25, 26-35, 36-45, 46-55, 56+ |
| `sender_state` | Category | 10 Indian states |
| `sender_bank` / `receiver_bank` | Category | 8 banks (SBI, HDFC, ICICI, ...) |
| `device_type` | Category | Android, iOS, Web |
| `network_type` | Category | 3G, 4G, 5G, WiFi |
| `hour_of_day` | Number | 0 to 23 (copied from the timestamp) |
| `day_of_week` | Category | Monday ... Sunday (copied from the timestamp) |
| `is_weekend` | 0/1 | 1 on Saturday and Sunday (copied from the timestamp) |
| **`fraud_flag`** | **Target** | **1 = fraud, 0 = legitimate. This is what we predict** |

**Data quality:** no missing values, no duplicate rows, no duplicate ids. The time columns
match the timestamp exactly. The data is clean, but it has **no user id or account id**, so we
cannot build "history" features (see [Limitations](#13-limitations-and-future-work)).

---

## 4. Workflow

```
 raw CSV
    │
    ▼
┌─────────────────┐   01_data_understanding.ipynb, 02_eda.ipynb
│ Understand + EDA │   (look at the data, draw graphs, find problems)
└────────┬────────┘
         ▼
┌─────────────────┐   ml/src/data_cleaning.py
│  Data cleaning   │──▶ data/processed/cleaned_data.csv
└────────┬────────┘
         ▼
┌─────────────────┐   ml/src/preprocessing.py, 03_data_preprocessing.ipynb
│  Preprocessing   │──▶ X_train, X_test, y_train, y_test (stratified 80/20 split)
└────────┬────────┘
         ▼
┌─────────────────┐   ml/src/feature_engineering.py, 04_feature_engineering.ipynb
│ Feature engineering│  5 new features + encoding + feature selection
└────────┬────────┘
         ▼
┌─────────────────┐   ml/src/train_model.py, 05_model_training.ipynb
│  Model training  │   4 models x 3 imbalance methods → tune the best one
└────────┬────────┘
         ▼
┌─────────────────┐   ml/artifacts/best_model.pkl + model_info.json
│  Saved pipeline  │   (features → encode → select → model, all in one file)
└────────┬────────┘
         ▼
┌─────────────────┐   ml/src/evaluate_model.py, 06, 07, 08 notebooks
│  Evaluation      │   metrics, curves, interpretability, prediction tests
└────────┬────────┘
         ▼
┌─────────────────┐   app/  (Streamlit, 6 pages)
│  Web app         │
└─────────────────┘
```

---

## 5. Folder structure: what every file does

```
UPI_Fraud_Detection/
├── app/                     Streamlit web app
├── data/                    data files
├── deployment/              Docker file
├── ml/                      machine learning code, notebooks, saved model
├── reports/                 graphs and result tables
├── tests/                   automatic tests
├── README.md                this file
├── requirements.txt         list of Python libraries
├── LICENSE                  license of the project
└── .gitignore               files Git must not track
```

### 5.1 `data/`

| File | What it is |
|---|---|
| `raw/upi_transactions_2024_raw.csv` | The original dataset. Never edited by hand |
| `processed/cleaned_data.csv` | Output of `data_cleaning.py`. Used by the app (Dashboard, Fraud Analytics) and by training |
| `processed/X_train.csv`, `X_test.csv` | Encoded and scaled input features (output of `preprocessing.py`) |
| `processed/y_train.csv`, `y_test.csv` | The target (`fraud_flag`) for train and test rows |
| `prediction_history.csv` | Created by the app. Every prediction made on the New Prediction page |
| `interim/`, `.gitkeep` files | Empty placeholders so Git keeps the folders |

### 5.2 `ml/src/`: the core Python code

| File | What it does |
|---|---|
| `data_cleaning.py` | Renames columns, handles missing values and duplicates, fixes data types, removes invalid rows, reports outliers, saves `cleaned_data.csv` and `reports/cleaning_summary.txt` |
| `preprocessing.py` | Splits X and y, one-hot encodes text columns, scales numbers, does a **stratified** 80/20 split, saves the four train/test files and `preprocessing_pipeline.pkl` |
| `feature_engineering.py` | Adds 5 new columns, builds the encoder, loads the same train/test split as `preprocessing.py` |
| `train_model.py` | Prepares data, compares 4 models × 3 imbalance methods, tunes the best, saves `best_model.pkl`, `model_info.json` and `reports/model_comparison.csv` |
| `evaluate_model.py` | Tests the saved model on the **test** data: metrics, ROC/PR curves, confusion matrix, `reports/evaluation_report.json` |
| `predict.py` | Loads the saved model and predicts for one transaction (`predict_one`) or many (`predict_many`), with a risk level (Low / Medium / High) |

### 5.3 `ml/notebooks/`: step-by-step notebooks

| Notebook | Purpose |
|---|---|
| `01_data_understanding.ipynb` | Shape, columns, types, missing values, duplicates, target, column classification |
| `02_eda.ipynb` | Exploratory data analysis: class imbalance, amount distribution, fraud rate by category, correlations, outliers, significance tests. Saves graphs to `reports/figures/` |
| `03_data_preprocessing.ipynb` | Runs the preprocessing steps one by one (X/y, split, encoding, scaling) |
| `04_feature_engineering.ipynb` | Shows the new features and checks if they relate to fraud |
| `05_model_training.ipynb` | Model comparison, tuning, saving, final check |
| `06_model_evaluation.ipynb` | Metrics, curves, confusion matrix, threshold table, probability histogram |
| `07_model_interpretability.ipynb` | Model coefficients and permutation importance |
| `08_prediction_testing.ipynb` | Tests the prediction functions: normal case, risky case, batch, unknown values, missing columns |

### 5.4 `ml/artifacts/`: saved model files

| File | What it is |
|---|---|
| `best_model.pkl` | **The final model**: one scikit-learn `Pipeline` (add features → encode → select 30 features → Logistic Regression). Accepts the 14 raw input columns |
| `model_info.json` | Model name, best settings, threshold, list of input columns, test results. The app reads this file |
| `preprocessing_pipeline.pkl` | The encoder from `preprocessing.py` (Person 1). The final model has its **own** encoder inside `best_model.pkl`, so this file is not needed by the app |

### 5.5 `reports/`

| File | What it is |
|---|---|
| `figures/01_…14_*.png` | EDA graphs (explained in [section 8](#8-the-graphs-explained)) |
| `figures/20_roc_pr_curves.png`, `21_confusion_matrix.png` | Evaluation graphs created by `evaluate_model.py` |
| `cleaning_summary.txt` | Before/after data quality summary |
| `model_comparison.csv` | Results of all 12 model/method combinations |
| `evaluation_report.json` | Final test metrics and confusion matrix numbers |

### 5.6 `app/`: the Streamlit app

| File | What it does |
|---|---|
| `streamlit_app.py` | Main page with links to all pages and the model warning |
| `config.py` | Paths, drop-down choices, warning text. Change settings here |
| `pages/1_Dashboard.py` | Overview of the dataset and of predictions made in the app |
| `pages/2_New_Prediction.py` | Check one transaction (form) or many (CSV upload) |
| `pages/3_Transaction_History.py` | Table of all predictions made, with a risk filter and download |
| `pages/4_Fraud_Analytics.py` | Fraud rate by any column and by amount range |
| `pages/5_Model_Performance.py` | Metrics, a threshold slider, confusion matrix, curves |
| `pages/6_Settings.py` | Change the threshold, clear the history, see model info |
| `utils/model_loader.py` | Loads the model and `model_info.json` |
| `utils/prediction.py` | Functions that build a transaction and call `predict.py` |
| `utils/data_loader.py` | Reads the cleaned data and saves/reads the prediction history |
| `utils/helpers.py` | Threshold in the session, number formatting |
| `components/sidebar.py` | Small info box in the sidebar |
| `components/metric_cards.py` | Rows of numbers (`st.metric`) |
| `components/charts.py` | Bar charts and tables |
| `components/prediction_result.py` | Shows the prediction result (probability bar + Low/Medium/High message) |

### 5.7 `tests/`, `deployment/` and root files

| File | What it does |
|---|---|
| `tests/test_data.py` | Checks the cleaned data (no nulls, target is 0/1, amount > 0) and the train/test files (no leakage, stratified) |
| `tests/test_model.py` | Checks the saved model loads and gives probabilities between 0 and 1 |
| `tests/test_prediction.py` | Checks the prediction functions (output format, missing columns, unknown categories, threshold) |
| `deployment/Dockerfile` | Instructions to run the app in a Docker container |
| `requirements.txt` | Python libraries needed |
| `.gitignore` | Files Git should ignore (cache folders, history file, ...) |

---

## 6. Step by step: what we did and why

### 6.1 Data understanding (notebook 01)
We load the file, look at the first and last rows, count rows and columns, check data types,
missing values and duplicates, and describe numeric and text columns. We identify:
`fraud_flag` is the **target**; `amount (INR)` and `hour_of_day` are **numeric**; 11 columns are
**categorical**; `is_weekend` is **binary**; `transaction id` is **irrelevant** (an id).

### 6.2 Exploratory data analysis (notebook 02)
Goal: find what is related to fraud. Main findings:

- Only **480 of 250,000** rows are fraud (0.192%). This is the biggest problem.
- Amounts are **right-skewed**: many small payments, few very large ones.
- Fraud and legitimate amounts are almost the same (median Rs 618 vs Rs 629).
- Fraud rate is about the same for every type, category, bank, state, device and hour.
- All correlations with fraud are near 0.
- Only small hints: amounts above Rs 3,558 have a slightly higher fraud rate
  (0.27% vs 0.18%), and sender age group 18-25 is a bit higher than 46-55.

We also ran **chi-square tests**. A p-value below 0.05 means "probably a real link".
Most columns were above 0.05. Only `sender_age_group` was below it (p = 0.026), which with
7 tests could easily be luck.

### 6.3 Data cleaning (`data_cleaning.py`)

| Decision | Reason |
|---|---|
| Rename columns to `snake_case` | Spaces and brackets cause bugs |
| Missing values: median for numbers, mode for text; drop rows with no label | Median is not affected by outliers. A row without `fraud_flag` is useless for training. (No missing values exist, but the function is reusable) |
| Remove duplicate ids and duplicate rows | Duplicates would count the same event twice |
| Fix types: timestamp → datetime, strip spaces in text | Dates must be dates |
| Remove invalid rows: amount ≤ 0 or > Rs 1,00,000 (UPI limit), `fraud_flag` not 0/1, hour not 0-23 | Impossible values are errors |
| Recompute hour/day/weekend from the timestamp | Guarantees consistency |
| **Outliers: reported, not removed** | 21,171 amounts (8.5%) are above Rs 3,558, but these are real purchases (Education, Shopping). Removing them could remove real fraud |
| **No column is dropped** | Dropping columns must be agreed by the team first |

On this dataset the script removed **0 rows**; the data was already clean.

### 6.4 Preprocessing (`preprocessing.py`)

- **Drop** `transaction_id` and `timestamp` (the time information is already in `hour_of_day`,
  `day_of_week` and `is_weekend`).
- **One-hot encoding** for text columns. Example: `device_type` becomes 3 columns
  (`Android`, `iOS`, `Web`) with 0/1. We use it because these categories have **no natural order**
  (label encoding 1, 2, 3 would wrongly suggest "Web > iOS"). `handle_unknown="ignore"` means a
  new category in the future does not crash the model.
- **StandardScaler** for numbers (mean 0, standard deviation 1). Logistic Regression is sensitive
  to the size of numbers, and a Rs 40,000 amount would otherwise dominate an hour value of 14.
- **Stratified 80/20 split** (`stratify=y`, `random_state=42`). With only 0.19% fraud, a normal
  random split could put too few fraud rows in the test set. Stratified keeps 0.192% fraud in both:
  384 fraud in train (200,000 rows) and 96 fraud in test (50,000 rows).
- **Fit on training data only.** The scaler learns the mean and standard deviation from the
  training rows only, then is applied to the test rows. Otherwise test information leaks into
  training (**data leakage**) and results look better than they really are.
- **No SMOTE here.** Oversampling is done later, only on training data.

### 6.5 Feature engineering (`feature_engineering.py`)

We add 5 columns. Each is made from the **same row only**, never from the target or other rows,
so there is no leakage.

| New feature | Idea |
|---|---|
| `log_amount` | `log(1 + amount)` makes the skewed amount smoother |
| `is_night` | Hour 0-5: fraud often happens at night |
| `is_high_amount` | Amount above Rs 3,500 (the EDA outlier limit) |
| `same_bank` | Sender bank = receiver bank |
| `same_age_group` | Sender and receiver have the same age group |

After encoding there are **74 features**. **Feature selection** with `SelectKBest(f_classif, k=30)`
keeps the 30 features with the strongest statistical link to fraud (ANOVA F-test). Fewer features
means a simpler model that overfits less. Like everything else, it is fit on training data only.

**What we could not do:** "transaction frequency per user" (e.g. 10 payments in 5 minutes), the
strongest signal in real fraud detection, because the data has **no user id**.

### 6.6 Handling class imbalance

With 0.19% fraud, a model that always says "legit" is 99.8% accurate and useless. We compared
three approaches on a validation split of the training data:

| Method | How it works | Result here |
|---|---|---|
| **nothing** | Train normally | Models predict "legit" for everything (precision, recall, F1 = 0) |
| **class weights** | Mistakes on fraud rows cost about 500× more (`class_weight="balanced"` / `scale_pos_weight`) | The model finds some fraud, but flags thousands of legit payments |
| **SMOTE** | Creates fake fraud rows by mixing real ones, until fraud is 10% of the legit size | Did not help; with so little signal it adds noisy fake rows |

**Important rule:** SMOTE is applied **only to the training part**, never to validation or test
data. The test set must show the real world, where fraud is rare.

### 6.7 Model comparison and tuning

1. **Validation comparison:** 4 models × 3 methods = 12 runs (`reports/model_comparison.csv`).
2. **Tuning:** the best model is tuned with `GridSearchCV` (3-fold cross-validation, scoring =
   PR-AUC) on **training data only**.
3. **Final test:** the test set is used **once**, at the end, to report results. We never use it to
   choose models or settings.

Validation results (sorted by PR-AUC; random guessing gives about 0.0019):

| Model | Method | ROC-AUC | PR-AUC |
|---|---|---|---|
| Logistic Regression | smote | 0.567 | 0.0043 |
| Logistic Regression | class_weight | 0.575 | 0.0039 |
| Logistic Regression | nothing | 0.563 | 0.0032 |
| Random Forest | nothing | 0.573 | 0.0026 |
| Decision Tree | nothing | 0.595 | 0.0025 |
| XGBoost | nothing | 0.523 | 0.0025 |
| … | … | 0.50-0.55 | 0.0020-0.0023 |

All values are close to random. The validation set has only about 77 fraud rows, so differences
are mostly noise. Logistic Regression was the top model, so it was tuned (best `C = 0.01`, a strong
regularization). The tuning step uses the class-weighted version (a simplification), and the final
model is Logistic Regression with class weights.

### 6.8 The saved model

`best_model.pkl` is one scikit-learn `Pipeline`:
`add features → encode and scale → select 30 features → Logistic Regression`.
Because everything is inside one file, the app only needs the 14 raw input columns. The
decision threshold (0.5) and the input column list are in `model_info.json`.

---

## 7. The algorithms and why we chose them

We choose a **range** of models, from simple to complex, to see whether complexity helps.

### Logistic Regression: the simple baseline (final choice)
- **How it works:** gives each feature a weight, adds them up and turns the sum into a
  probability between 0 and 1 (S-shaped curve).
- **Why we use it:** fast, gives real probabilities, and its weights can be read
  (positive weight = more fraud). With `C` small, it is strongly regularized and does not
  overfit noise.
- **Weakness:** can only learn straight-line relationships.
- **Why it won here:** the data has almost no signal, so a complex model can only memorize
  noise. The simplest, most regularized model generalizes best (but only marginally).

### Decision Tree: readable rules
- **How it works:** asks a chain of yes/no questions ("amount > 3500?").
- **Why we use it:** easy to understand and can catch non-linear rules.
- **Weakness:** one deep tree memorizes the training data. We limit it with `max_depth` and
  `min_samples_leaf`.

### Random Forest: many trees voting
- **How it works:** trains many trees on random parts of the data and features, then averages them.
- **Why we use it:** more stable than one tree, handles feature interactions, and needs
  little tuning.
- **Weakness:** slower; can overfit very rare classes.

### XGBoost: boosted trees
- **How it works:** builds trees one after another; each new tree fixes the errors of the previous ones.
- **Why we use it:** one of the best methods for tabular data in fraud detection. `scale_pos_weight`
  handles imbalance directly.
- **Weakness:** many settings; easy to overfit when there are only 384 fraud rows.

### Why not deep learning or anomaly detection?
Only 384 training fraud rows is far too few for a neural network. Anomaly detection (Isolation
Forest) assumes fraud looks **different** from normal payments. Our EDA shows it does not.

---

## 8. The graphs explained

How to read the main plots, and what they show for **our** data.

### Metrics we use (and why not accuracy)

| Metric | Meaning | Why |
|---|---|---|
| **Precision** | Of the transactions we flagged, how many were really fraud | Low precision = many false alarms |
| **Recall** | Of all real fraud, how many we caught | Low recall = missed fraud |
| **F1-score** | Balance of precision and recall | One number for both |
| **ROC-AUC** | Chance that a random fraud gets a higher score than a random legit one. 0.5 = guessing, 1.0 = perfect | Does not depend on the threshold |
| **PR-AUC** | Area under the precision-recall curve. Random guessing equals the fraud rate (0.0019) | Best for very imbalanced data |
| **Accuracy** | Share of correct predictions | **Not used.** "Always legit" scores 99.8% |

### EDA graphs (`reports/figures/`)

| Graph | How to read it | What it shows |
|---|---|---|
| `01_class_imbalance.png` | Left: bar counts. Right: same data on a log scale | The fraud bar is invisible: 480 vs 249,520 |
| `02_amount_distribution.png` | Histogram of amount, histogram of log(amount), boxplot by class | Amounts are right-skewed; log makes them bell-shaped; fraud and legit boxes look the same |
| `03…10_fraud_by_*.png` | Bars = fraud rate (%) per group. **Red dashed line = overall rate** | Bars near the red line = the group does not matter. Nearly all bars are near it. Recharge (0.24%) and WiFi (0.24%) are slightly above, but based on few cases |
| `11_fraud_by_hour.png` | Fraud rate per hour | Jumps between 0.06% and 0.30% with no clear night pattern: this is noise |
| `12_fraud_by_month.png` | Fraud rate per month | No seasonal pattern |
| `13_correlation_heatmap.png` | Colors = correlation between numeric columns (−1 to +1). Red/blue = strong, white = none | All values near 0 (fraud vs amount ≈ 0.004) |
| `14_amount_fraud_vs_legit.png` | Two density curves of log(amount) | The curves nearly overlap: amount alone cannot separate fraud |

### Evaluation graphs

| Graph | How to read it | What it shows |
|---|---|---|
| `20_roc_pr_curves.png` (left, ROC) | X: false-positive rate, Y: true-positive rate. **Dashed diagonal = random guessing.** A good model bends to the top-left | Our curve hugs the diagonal (AUC 0.49) |
| `20_roc_pr_curves.png` (right, PR curve) | X: recall, Y: precision. **Red line = random (0.0019).** A good model stays high | Our curve is flat near the red line |
| `21_confusion_matrix.png` | 2×2 table: rows = actual, columns = predicted | At threshold 0.5: 38 fraud caught, 58 missed, **20,399 legit flagged**, 29,505 correct |

### Charts inside notebooks
- **Model comparison bar chart (notebook 05):** PR-AUC per model and method. Bars are about as high as the red random line.
- **Threshold table (notebook 06):** moving the threshold only trades recall for false alarms; precision stays about 0.19% (the fraud rate) at every threshold, which means the model's ranking is no better than random.

| Threshold | Fraud caught (recall) | Transactions flagged |
|---|---|---|
| 0.3 | 96.9% | 48,550 of 50,000 |
| 0.4 | 80.2% | 39,508 |
| 0.5 | 39.6% | 20,437 |
| 0.6 | 8.3% | 5,674 |

- **Probability histogram (notebook 06):** the legit and fraud histograms overlap almost completely.
- **Coefficients and permutation importance (notebook 07):** coefficients are small and point in many directions, and `is_high_amount` is the largest "more fraud" feature. Permutation importance (the score drop when a column is shuffled) is tiny and often **negative**, meaning shuffling that column did not hurt, so it carries no useful signal.

---

## 9. Why the model is weak

1. **The features have no link to fraud.** Every graph and test in the EDA points to this.
2. **Very few fraud rows** (384 for training), so a model can easily learn noise.
3. **No user / account / device id**, so no history-based features.
4. **The labels look synthetic**, since the fraud flag seems unrelated to the other columns.

**A warning sign to remember:** if anyone reports ROC-AUC above 0.9 on this data, suspect
**data leakage** (target information reaching the features) or an evaluation mistake, not a great model.

**What would help:** a user id and transaction history, real fraud labels from investigations,
more fraud examples, and features such as "number of transactions in the last hour".

---

## 10. The Streamlit app

Start it with `python -m streamlit run app/streamlit_app.py`. Pages:

| Page | What you can do |
|---|---|
| **Dashboard** | Dataset numbers (transactions, fraud, average amount), transactions and fraud per month, predictions made in the app |
| **New Prediction** | Enter one transaction and see the fraud probability and a Low / Medium / High message. A second tab checks a whole CSV file and lets you download the results |
| **Transaction History** | Every prediction made in the app, filtered by risk level, with download |
| **Fraud Analytics** | Choose a column and see the fraud rate per value, plus fraud rate by amount range |
| **Model Performance** | Metrics, a threshold slider showing fraud caught vs false alarms on the test data, confusion matrix, ROC/PR curves |
| **Settings** | Change the fraud threshold (for the session), clear the history, see model information |

**Risk levels:** High = probability ≥ threshold (flagged as fraud); Medium = at least 80% of the
threshold; Low = below that.

---

## 11. Tests

```bash
python -m pytest tests
```
18 tests check that the data is clean (no nulls, target is 0/1), the train/test files have no
leakage and keep the same fraud rate, the model loads and returns probabilities between 0 and 1,
`predict_one` / `predict_many` return the right format, missing columns give a clear error, unknown
categories do not crash, and changing the threshold changes the prediction.
Tests that need a file (e.g. `best_model.pkl`) are skipped with a message if it is missing.

---

## 12. Deployment with Docker

```bash
docker build -f deployment/Dockerfile -t upi-fraud .
docker run -p 8501:8501 upi-fraud
```
Then open `http://localhost:8501`. The model file `ml/artifacts/best_model.pkl` must exist
before building the image.

---

## 13. Limitations and future work

- The model is not better than random guessing; do not use it for real decisions.
- The model was trained once on one year of (probably synthetic) data.
- No transaction history features (no user id).
- Tuning is small (a few settings) to stay fast.
- The fraud threshold is fixed at 0.5. With such weak signal there is no good threshold.
- Prediction history is a simple CSV file, not a database.

**Ideas:** add a user id and rolling features, get real labeled fraud data, try cost-based
thresholds (a missed fraud costs more than a false alarm), and monitor the model after deployment.

---

## 14. Team and Git workflow

| Person | Work |
|---|---|
| Person 1 | Data understanding, EDA, cleaning, preprocessing |
| Person 2 | Feature engineering, class imbalance, model training and tuning |
| Person 3 | Evaluation, interpretability, prediction testing, Streamlit app |

- Work on a branch (`feature/...`), never commit directly to `main`.
- Commit small steps with clear messages, push, and open a **Pull Request** into `main`.
- Run `git pull origin main` before starting work, so you are not behind.
- `ml/artifacts/best_model.pkl` is blocked by default by `.gitignore` (`*.pkl`). To share the
  model, allow it with the line `!ml/artifacts/best_model.pkl` and `git add` it.

---

## 15. Glossary

| Term | Simple meaning |
|---|---|
| **Target** | The column we predict (`fraud_flag`) |
| **Feature** | An input column the model uses |
| **Class imbalance** | One class (fraud) is much rarer than the other |
| **Data leakage** | Information from the test set or the target sneaks into training, making results look too good |
| **Stratified split** | A split that keeps the same fraud percentage in train and test |
| **One-hot encoding** | Turn a text column into several 0/1 columns |
| **Scaling** | Put numbers on a similar size (mean 0, standard deviation 1) |
| **SMOTE** | Makes synthetic examples of the rare class |
| **Class weight** | Makes mistakes on the rare class cost more |
| **Overfitting** | The model memorizes training data and fails on new data |
| **Cross-validation** | Split the training data into folds; train on some, check on the rest, repeat |
| **Hyperparameter** | A setting chosen before training (e.g. tree depth) |
| **Threshold** | The probability above which we call a transaction fraud |
| **Pipeline** | Several steps chained into one object |
| **Permutation importance** | How much the score drops when one column is shuffled |

---

## 16. FAQ

**Why is accuracy not shown?** Always predicting "legit" gives 99.8%, so accuracy hides failures.

**Why is the test set used only once?** If we pick models using the test set, the final score is
too optimistic. Choices are made with validation and cross-validation on training data.

**Why did a simple model win?** With almost no signal, complex models fit noise. Simple, strongly
regularized models are safest.

**The app says `cleaned_data.csv not found`.** Run `python ml/src/data_cleaning.py`.

**The app says the model file is missing.** Run `python ml/src/train_model.py`, or get
`best_model.pkl` from a teammate.

**`No module named feature_engineering` when loading the model.** Add `ml/src` to `sys.path`
before `joblib.load`, because the saved pipeline refers to `add_features` in that file.

**Results are different on my computer.** Small differences are normal with other library versions.
Use the same scikit-learn and XGBoost versions as the teammate who trained the model.