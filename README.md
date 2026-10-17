# SKYC — Smart KYC-Based Financial Risk & Suspicious Activity Detection

Graduation project (B.Sc. Data Science, Al-Balqa Applied University, 2025/2026) — an ML system that scores transactions and customers for money-laundering and fraud risk, with rule-based explanations and a Streamlit dashboard.

**Team:** Najd Raid Hwarat · Hala Nedal Amoor · Joud Salah Saleh — **Supervisor:** Dr. Nisrean Thalji

📄 Full project report: [docs/SKYC_report.pdf](docs/SKYC_report.pdf)

## What it does

- Trains three supervised models — Logistic Regression, Random Forest and XGBoost — on three datasets: **IBM AML** (primary), **Credit Card Fraud** and **PaySim**.
- Averages the three model probabilities into a **transaction risk score (0–100)** and aggregates it per account into a **customer risk score**:
  `0.55 × average tx risk + 0.20 × highest tx risk + 0.25 × % of tx scoring ≥ 60`.
- Classifies risk as LOW (0–30), MEDIUM (30–60), HIGH (60–85) or CRITICAL (85–100) and explains alerts with rules built on 60+ engineered behavioural features (cross-bank transfers, currency mismatch, spikes, off-hours activity, …).
- Presents everything in a 5-step Streamlit app: dataset → account → risk assessment → analytics dashboard → model performance.

## Repository layout

```
app/                    Streamlit application (ready to run)
  app.py
  requirements.txt      pinned versions used to train and save the models
  run.sh / run.bat      create a virtualenv, install, and start the app
  data/                 30,000-row feature samples used by the app (Parquet)
  models/               saved scalers, models (joblib) and feature lists (JSON)
notebooks/
  01_ibm_aml_pipeline.ipynb               IBM AML: sampling, feature engineering, training, K-Fold
  02_creditcard_paysim_fe_no_smote.ipynb  Credit Card + PaySim (adopted FE + No SMOTE config) and cross-dataset comparison
  experiments/                            the other three FE × SMOTE configurations
results/                figures and CSV results produced by the notebooks
docs/SKYC_report.pdf    graduation project report
requirements-notebooks.txt
```

## Run the app

```bash
cd app
./run.sh          # macOS / Linux  (Windows: run.bat)
```

or manually:

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

The app opens at http://localhost:8501. All data and models it needs are included in `app/`.

## Results (fraud class, 20% stratified hold-out test split)

| Dataset | Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| IBM AML | Logistic Regression | 0.8372 | 0.8330 | 0.8351 | 0.9035 | 0.8132 |
| IBM AML | Random Forest | 0.8460 | 0.8325 | 0.8392 | 0.9119 | 0.8631 |
| IBM AML | XGBoost | 0.8867 | 0.8180 | 0.8510 | 0.9191 | 0.8886 |
| Credit Card | Logistic Regression | 0.4335 | 0.8980 | 0.5847 | 0.9820 | 0.9081 |
| Credit Card | Random Forest | 0.9659 | 0.8673 | 0.9140 | 0.9856 | 0.9208 |
| Credit Card | XGBoost | 1.0000 | 0.8265 | 0.9050 | 0.9862 | 0.9256 |
| PaySim | Logistic Regression | 0.9244 | 0.9458 | 0.9350 | 0.9933 | 0.9863 |
| PaySim | Random Forest | 0.9988 | 0.9970 | 0.9979 | 0.9997 | 0.9995 |
| PaySim | XGBoost | 0.9988 | 0.9963 | 0.9976 | 0.9995 | 0.9991 |

IBM AML 5-fold cross-validation ROC-AUC: XGBoost 0.9237 ± 0.0020, Random Forest 0.9168 ± 0.0053, Logistic Regression 0.9091 ± 0.0025.

### Read these numbers with care

- The working samples contain far more fraud than reality: **IBM AML 33.3%** (real 0.09%), **PaySim 27.4%** (real 0.13%), **Credit Card 1.64%** (real 0.17%). Precision, F1 and PR-AUC would be much lower at the real fraud rate, and thresholds must be recalibrated before any production use.
- On IBM AML the models rely heavily on the payment format (`Payment Format_ACH` ≈ 54% of Random Forest importance).
- PaySim is a simulator with simple fraud rules, so its near-perfect scores mostly reflect those rules.
- Engineered features are used only for the dashboard and explanations, not as model inputs, to avoid temporal and sampling leakage.

## Data

The raw datasets are too large for GitHub (≈10 GB in total). Download them from Kaggle:

| Dataset | Source | Used by |
|---|---|---|
| IBM Transactions for Anti-Money Laundering (HI-Large / LI-Large transactions and accounts) | [kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) | notebook 01 |
| Credit Card Fraud Detection | [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | notebook 02 |
| PaySim synthetic financial dataset | [kaggle.com/datasets/ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1) | notebook 02 |

Save the files in `data/raw/`: `creditcard.csv` and `paysim.csv` for notebook 02 and the experiments, and the four IBM AML Large files (about 40 GB) for notebook 01.

```bash
pip install -r requirements-notebooks.txt
jupyter notebook notebooks/
```

## Tech stack

Python 3.12 · pandas · scikit-learn 1.9.0 · XGBoost 3.2.0 · imbalanced-learn (SMOTE) · Streamlit · Plotly · Parquet (pyarrow) · joblib
