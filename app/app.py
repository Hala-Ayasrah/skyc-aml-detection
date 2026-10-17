"""
SKYC — Smart KYC-Based Financial Risk Detection Platform
=========================================================
3 datasets: IBM AML Transactions, Credit Card Fraud, PaySim
9 models: LR + RF + XGB per dataset
Step-based navigation for compliance officers
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
DATA = BASE / "data"

DATASETS = {
    "AML Transactions": {
        "desc": "IBM AML banking transaction flow analysis",
        "parquet": DATA / "df_merged_features.parquet",
        "models": {
            "lr": MODELS / "lr_aml_model.pkl",
            "rf": MODELS / "rf_aml_model.pkl",
            "xgb": MODELS / "xgb_aml_model.pkl",
        },
        "scaler": MODELS / "scaler_aml.pkl",
        "features_json": MODELS / "aml_features.json",
        "account_col": "Account",
        "target_col": "Is Laundering",
        "amount_col": "Amount Paid",
        "timestamp_col": "Timestamp",
    },
    "Credit Card": {
        "desc": "Credit card purchase behaviors and fraud patterns",
        "parquet": DATA / "cc_features.parquet",
        "models": {
            "lr": MODELS / "lr_creditcard_no_smote.pkl",
            "rf": MODELS / "rf_creditcard_no_smote.pkl",
            "xgb": MODELS / "xgb_creditcard_no_smote.pkl",
        },
        "scaler": MODELS / "scaler_creditcard.pkl",
        "features_json": MODELS / "cc_features.json",
        "account_col": "Account_ID",
        "target_col": "Class",
        "amount_col": "Amount",
        "timestamp_col": None,
    },
    "PaySim": {
        "desc": "Mobile money transaction simulation analysis",
        "parquet": DATA / "ps_features.parquet",
        "models": {
            "lr": MODELS / "lr_paysim_no_smote.pkl",
            "rf": MODELS / "rf_paysim_no_smote.pkl",
            "xgb": MODELS / "xgb_paysim_no_smote.pkl",
        },
        "scaler": MODELS / "scaler_paysim.pkl",
        "features_json": MODELS / "paysim_features.json",
        "account_col": "nameOrig",
        "target_col": "isFraud",
        "amount_col": "amount",
        "timestamp_col": None,
    },
}

RISK_COLORS = {
    "LOW": "#2D6A4F", "MEDIUM": "#B8860B",
    "HIGH": "#9A3412", "CRITICAL": "#7F1D1D",
}

# Test-set (20% stratified hold-out) metrics of the saved models, fraud class.
# Source: results/ibm_aml/model_comparison.csv and results/creditcard_paysim/*_results.csv
MODEL_PERF = {
    "AML Transactions": {
        "LR":  {"Precision": 0.8372, "Recall": 0.8330, "F1": 0.8351, "ROC-AUC": 0.9035, "PR-AUC": 0.8132},
        "RF":  {"Precision": 0.8460, "Recall": 0.8325, "F1": 0.8392, "ROC-AUC": 0.9119, "PR-AUC": 0.8631},
        "XGB": {"Precision": 0.8867, "Recall": 0.8180, "F1": 0.8510, "ROC-AUC": 0.9191, "PR-AUC": 0.8886},
    },
    "Credit Card": {
        "LR":  {"Precision": 0.4335, "Recall": 0.8980, "F1": 0.5847, "ROC-AUC": 0.9820, "PR-AUC": 0.9081},
        "RF":  {"Precision": 0.9659, "Recall": 0.8673, "F1": 0.9140, "ROC-AUC": 0.9856, "PR-AUC": 0.9208},
        "XGB": {"Precision": 1.0000, "Recall": 0.8265, "F1": 0.9050, "ROC-AUC": 0.9862, "PR-AUC": 0.9256},
    },
    "PaySim": {
        "LR":  {"Precision": 0.9244, "Recall": 0.9458, "F1": 0.9350, "ROC-AUC": 0.9933, "PR-AUC": 0.9863},
        "RF":  {"Precision": 0.9988, "Recall": 0.9970, "F1": 0.9979, "ROC-AUC": 0.9997, "PR-AUC": 0.9995},
        "XGB": {"Precision": 0.9988, "Recall": 0.9963, "F1": 0.9976, "ROC-AUC": 0.9995, "PR-AUC": 0.9991},
    },
}


# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG + CSS
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="SKYC AML Platform", layout="wide")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0D1117; color: #E6EDF3; }
#MainMenu, footer, header { visibility: hidden; }
.top-bar {
    background: linear-gradient(135deg, #0B2545 0%, #13315C 100%);
    color: white; padding: 1.2rem 2rem; margin: -1rem -1rem 1.5rem -1rem;
    border-bottom: 3px solid #1E5F9F;
}
.top-bar h1 { color: white; font-size: 1.5rem; font-weight: 700; margin: 0; }
.top-bar .sub { color: #7EB3E0; font-size: 0.78rem; letter-spacing: 0.08em;
    text-transform: uppercase; margin-top: 0.15rem; }
.kpi { background: #161B22; padding: 1.1rem 1.3rem; border-radius: 8px;
    border: 1px solid #30363D; border-left: 4px solid #1E5F9F; }
.kpi-label { color: #8B949E; font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.07em; margin: 0; }
.kpi-val { color: #E6EDF3; font-size: 1.7rem; font-weight: 700;
    margin: 0.3rem 0 0.15rem; line-height: 1; }
.kpi-sub { color: #8B949E; font-size: 0.75rem; margin: 0; }
.risk-box { padding: 1.3rem; border-radius: 8px; border: 1px solid #30363D;
    background: #161B22; text-align: center; }
.risk-score { font-size: 2.5rem; font-weight: 700; color: #E6EDF3; line-height: 1; }
.risk-label { font-size: 0.8rem; color: #8B949E; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.07em; }
.badge { display: inline-block; padding: 0.25rem 0.7rem; border-radius: 4px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; color: white; }
.reason-card { background: #161B22; padding: 0.85rem 1.1rem; border-radius: 6px;
    border: 1px solid #30363D; border-left: 3px solid #1E5F9F; margin-bottom: 0.45rem; }
.reason-cat { font-size: 0.68rem; font-weight: 700; color: #8B949E;
    text-transform: uppercase; letter-spacing: 0.07em; }
.reason-msg { font-size: 0.85rem; color: #E6EDF3; margin-top: 0.15rem; }
.section-hdr { color: #7EB3E0; font-size: 0.95rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.07em;
    margin: 1.3rem 0 0.8rem; padding-bottom: 0.4rem;
    border-bottom: 2px solid #30363D; }
</style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def classify_risk(score):
    if score < 30: return "LOW"
    if score < 60: return "MEDIUM"
    if score < 85: return "HIGH"
    return "CRITICAL"

def kpi(label, value, sub=""):
    return f'<div class="kpi"><p class="kpi-label">{label}</p><p class="kpi-val">{value}</p><p class="kpi-sub">{sub}</p></div>'

def risk_panel(score, label="RISK SCORE"):
    level = classify_risk(score)
    color = RISK_COLORS[level]
    return f'''<div class="risk-box" style="border-left:5px solid {color};">
        <p class="risk-label">{label}</p>
        <p class="risk-score">{score:.1f}</p>
        <span class="badge" style="background:{color}">{level}</span></div>'''

def reason_card(cat, level, msg):
    color = RISK_COLORS.get(level, "#1E5F9F")
    return f'''<div class="reason-card" style="border-left-color:{color};">
        <div class="reason-cat">{cat} - {level}</div>
        <div class="reason-msg">{msg}</div></div>'''

def section(title):
    return f'<div class="section-hdr">{title}</div>'

def plotly_layout():
    return dict(plot_bgcolor="#161B22", paper_bgcolor="#161B22",
                font=dict(family="DM Sans, Arial", color="#E6EDF3", size=12),
                xaxis=dict(gridcolor="#30363D"), yaxis=dict(gridcolor="#30363D"),
                colorway=["#0B2545","#1E5F9F","#3B82B5","#5A6478","#B8860B","#9A3412","#2D6A4F"],
                margin=dict(l=40,r=20,t=50,b=40))


# ═══════════════════════════════════════════════════════════════
# DATA + MODEL LOADING
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_models(ds_name):
    cfg = DATASETS[ds_name]
    models = {}
    for key in ["lr", "rf", "xgb"]:
        p = cfg["models"][key]
        if p.exists():
            m = joblib.load(p)
            if not hasattr(m, 'multi_class'):
                try: m.multi_class = 'auto'
                except: pass
            models[key] = m
    scaler = joblib.load(cfg["scaler"]) if cfg["scaler"].exists() else None
    features = []
    if cfg["features_json"].exists():
        with open(cfg["features_json"]) as f:
            features = json.load(f)
    return models, scaler, features

@st.cache_data
def load_data(ds_name):
    cfg = DATASETS[ds_name]
    if not cfg["parquet"].exists():
        return None
    df = pd.read_parquet(cfg["parquet"])
    if cfg["timestamp_col"] and cfg["timestamp_col"] in df.columns:
        df[cfg["timestamp_col"]] = pd.to_datetime(df[cfg["timestamp_col"]], errors="coerce")
    # Add OHE columns if missing (parquet saved before OHE)
    ohe_source = ["Payment Format", "Payment Currency", "Receiving Currency"]
    ohe_present = [c for c in ohe_source if c in df.columns]
    if ohe_present:
        dummies = pd.get_dummies(df[ohe_present], drop_first=False).astype(int)
        for col in dummies.columns:
            if col not in df.columns:
                df[col] = dummies[col]
    return df

@st.cache_data
def score_transactions(_df, _models, _scaler, _features, ds_name):
    feat_cols = [c for c in _features if c in _df.columns]
    if not feat_cols or not _models or not _scaler:
        return _df
    X = _df[feat_cols].fillna(0).astype(float).values
    X_sc = _scaler.transform(X)
    out = _df.copy()
    probas = {}
    for key in ["lr", "rf", "xgb"]:
        if key in _models:
            probas[key] = _models[key].predict_proba(X_sc)[:, 1]
            out[f"risk_{key}"] = (probas[key] * 100).round(2)
    if probas:
        avg = np.mean(list(probas.values()), axis=0)
        out["risk_score"] = (avg * 100).round(2)
        out["risk_level"] = out["risk_score"].apply(classify_risk)
    return out

def compute_customer_risk(df, cfg):
    acc_col = cfg["account_col"]
    target_col = cfg["target_col"]
    amount_col = cfg["amount_col"]
    if not acc_col or acc_col not in df.columns or "risk_score" not in df.columns:
        return pd.DataFrame()
    agg_dict = {
        "n_tx": (target_col, "count"),
        "n_fraud": (target_col, "sum"),
        "avg_risk": ("risk_score", "mean"),
        "max_risk": ("risk_score", "max"),
        "n_high": ("risk_score", lambda x: int((x >= 60).sum())),
    }
    if amount_col in df.columns:
        agg_dict["total_amount"] = (amount_col, "sum")
        agg_dict["avg_amount"] = (amount_col, "mean")
    agg = df.groupby(acc_col).agg(**agg_dict).reset_index()
    agg["high_ratio"] = (agg["n_high"] / agg["n_tx"].clip(lower=1) * 100).round(1)
    agg["fraud_rate"] = (agg["n_fraud"] / agg["n_tx"] * 100).round(2)
    # Customer risk = 55% average tx risk + 20% highest tx risk + 25% share of tx scoring >= 60
    agg["cust_risk"] = (
        0.55 * agg["avg_risk"] + 0.20 * agg["max_risk"] +
        0.25 * agg["high_ratio"]
    ).round(2).clip(0, 100)
    agg["cust_level"] = agg["cust_risk"].apply(classify_risk)
    return agg

def explain_tx(row, ds_name):
    reasons = []
    if ds_name == "AML Transactions":
        if row.get("is_cross_bank", 0) == 1:
            reasons.append(("Counterparty", "MEDIUM", "Cross-bank transfer detected"))
        if row.get("currency_mismatch", 0) == 1:
            reasons.append(("Currency", "MEDIUM", "Payment and receiving currencies differ"))
        if row.get("amount_spike", 0) == 1:
            reasons.append(("Behavior", "HIGH", "Monthly amount spike detected"))
        if row.get("tx_count_spike", 0) == 1:
            reasons.append(("Behavior", "HIGH", "Monthly transaction-count spike"))
        if row.get("receiver_spike", 0) == 1:
            reasons.append(("Behavior", "HIGH", "Unusual increase in new receivers"))
        if row.get("is_large_amount", 0) == 1:
            reasons.append(("Amount", "HIGH", "Amount exceeds account 95th percentile"))
        if row.get("is_faster_than_normal", 0) == 1:
            reasons.append(("Velocity", "MEDIUM", "Faster than account normal pace"))
        hr = row.get("hour", -1)
        if 0 <= hr <= 4:
            reasons.append(("Timing", "MEDIUM", f"Off-hours transaction (hour {int(hr):02d}:00)"))
        ratio = row.get("amount_to_mean_ratio", 1)
        if pd.notna(ratio) and ratio > 3:
            reasons.append(("Amount", "HIGH", f"Amount is {ratio:.1f}x account average"))
    elif ds_name == "PaySim":
        if row.get("amount_vs_balance", 0) > 0.9:
            reasons.append(("Balance", "CRITICAL", "Transaction uses >90% of available balance"))
        if row.get("balance_diff_orig", 0) > 0 and row.get("amount", 0) > 500000:
            reasons.append(("Amount", "HIGH", "Large transfer with significant balance change"))
        if row.get("dest_is_customer", 0) == 1:
            reasons.append(("Counterparty", "MEDIUM", "Destination is a customer account"))
    elif ds_name == "Credit Card":
        amt = row.get("Amount", 0)
        if amt > 1000:
            reasons.append(("Amount", "HIGH", f"High transaction amount: ${amt:,.2f}"))
        for v in ["V14", "V17", "V12", "V10"]:
            val = row.get(v, 0)
            if abs(val) > 3:
                reasons.append(("PCA Feature", "MEDIUM", f"{v} deviation: {val:.2f} (>3 std)"))
                break
    score = row.get("risk_score", 0)
    if not reasons and score >= 60:
        reasons.append(("Model", "HIGH", "ML ensemble detected suspicious patterns"))
    if not reasons:
        reasons.append(("Model", "LOW", "No anomaly indicators detected"))
    return reasons


# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
for k, v in {"step": 1, "dataset": "", "account": ""}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def go_to(step, **kw):
    st.session_state.step = step
    for k, v in kw.items():
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════
# TOP BAR
# ═══════════════════════════════════════════════════════════════
st.markdown('''<div class="top-bar">
    <h1>SKYC Anti-Money Laundering Detection Platform</h1>
    <div class="sub">Smart KYC-Based Financial Risk Detection</div>
</div>''', unsafe_allow_html=True)

step_labels = {1: "Select Dataset", 2: "Select Account", 3: "Risk Assessment",
               4: "Analytics Dashboard", 5: "Model Performance"}
st.progress(st.session_state.step / 5.0)
st.caption(f"Step {st.session_state.step} of 5: {step_labels[st.session_state.step]}")


# ═══════════════════════════════════════════════════════════════
# STEP 1 — DATASET SELECTION
# ═══════════════════════════════════════════════════════════════
if st.session_state.step == 1:
    st.markdown(section("Select Dataset Engine"), unsafe_allow_html=True)
    st.write("Choose the dataset to initialize the risk scoring pipeline.")

    c1, c2, c3 = st.columns(3)
    for col, (ds_name, cfg) in zip([c1, c2, c3], DATASETS.items()):
        with col:
            exists = cfg["parquet"].exists()
            status = "Ready" if exists else "Not found"
            st.markdown(f"""<div style="background:#161B22;border:1px solid #30363D;
                border-radius:8px;padding:1.2rem;min-height:160px;">
                <p style="font-weight:700;font-size:1rem;color:#E6EDF3;margin:0 0 0.5rem;">{ds_name}</p>
                <p style="font-size:0.85rem;color:#8B949E;margin:0 0 0.8rem;">{cfg['desc']}</p>
                <p style="font-size:0.75rem;color:{'#2D6A4F' if exists else '#9A3412'};">
                Status: {status}</p></div>""", unsafe_allow_html=True)
            if st.button(f"Select {ds_name}", key=f"sel_{ds_name}", width="stretch"):
                go_to(2, dataset=ds_name)
                st.rerun()


# ═══════════════════════════════════════════════════════════════
# STEP 2 — ACCOUNT SELECTION
# ═══════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    ds = st.session_state.dataset
    cfg = DATASETS[ds]
    st.markdown(section(f"Select Account -- {ds}"), unsafe_allow_html=True)

    df = load_data(ds)
    if df is None:
        st.error(f"Data file not found: {cfg['parquet']}")
        if st.button("Back"): go_to(1); st.rerun()
        st.stop()

    acc_col = cfg["account_col"]
    if acc_col and acc_col in df.columns:
        accounts = sorted(df[acc_col].astype(str).unique())
        account = st.selectbox(f"Select account ({len(accounts):,} available)", accounts)
    else:
        account = st.number_input("Enter transaction index", 0, len(df)-1, 0)
        account = str(account)

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Back"): go_to(1); st.rerun()
    with c2:
        if st.button("Analyze Risk", type="primary"):
            go_to(3, account=account)
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# STEP 3 — RISK ASSESSMENT
# ═══════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    ds = st.session_state.dataset
    cfg = DATASETS[ds]
    acc = st.session_state.account
    acc_col = cfg["account_col"]
    target_col = cfg["target_col"]
    amount_col = cfg["amount_col"]

    st.markdown(section(f"Risk Assessment -- {ds}"), unsafe_allow_html=True)

    df = load_data(ds)
    models, scaler, features = load_models(ds)
    df_scored = score_transactions(df, models, scaler, features, ds)

    if acc_col and acc_col in df_scored.columns:
        cust_txs = df_scored[df_scored[acc_col].astype(str) == str(acc)].copy()
    else:
        idx = int(acc)
        cust_txs = df_scored.iloc[max(0, idx-50):idx+50].copy()

    if len(cust_txs) == 0:
        st.warning("No transactions found for this account.")
        if st.button("Back"): go_to(2); st.rerun()
        st.stop()

    cust_risk_df = compute_customer_risk(df_scored, cfg)
    if acc_col and len(cust_risk_df) > 0:
        cust_row = cust_risk_df[cust_risk_df[acc_col].astype(str) == str(acc)]
        if len(cust_row) > 0:
            cust_row = cust_row.iloc[0]
            cust_score = cust_row["cust_risk"]
            n_tx = int(cust_row["n_tx"])
            fraud_rate = cust_row["fraud_rate"]
            n_high = int(cust_row["n_high"])
        else:
            cust_score = cust_txs["risk_score"].mean() if "risk_score" in cust_txs else 0
            n_tx = len(cust_txs)
            fraud_rate = (cust_txs[target_col].sum() / n_tx * 100) if target_col in cust_txs else 0
            n_high = (cust_txs["risk_score"] >= 60).sum() if "risk_score" in cust_txs else 0
    else:
        cust_score = cust_txs["risk_score"].mean() if "risk_score" in cust_txs else 0
        n_tx = len(cust_txs)
        fraud_rate = (cust_txs[target_col].sum() / max(n_tx, 1) * 100) if target_col in cust_txs else 0
        n_high = (cust_txs["risk_score"] >= 60).sum() if "risk_score" in cust_txs else 0

    tx_score = cust_txs["risk_score"].max() if "risk_score" in cust_txs else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(risk_panel(cust_score, "CUSTOMER RISK"), unsafe_allow_html=True)
    with c2:
        st.markdown(risk_panel(tx_score, "HIGHEST TX RISK"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Total Transactions", f"{n_tx:,}", f"{n_high} high-risk"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Fraud Rate", f"{fraud_rate:.1f}%",
            f"{int(cust_txs[target_col].sum())} confirmed" if target_col in cust_txs else ""), unsafe_allow_html=True)

    st.markdown(section("Risk Alerts -- Explainable AI"), unsafe_allow_html=True)

    if cust_score >= 85:
        st.error("CRITICAL: Immediate compliance review required. Multiple high-risk indicators detected.")
    elif cust_score >= 60:
        st.warning("HIGH RISK: Enhanced due diligence recommended. Suspicious patterns identified.")
    elif cust_score >= 30:
        st.info("MEDIUM RISK: Routine monitoring advised. Some unusual patterns observed.")
    else:
        st.success("LOW RISK: No suspicious activity detected. Account within normal parameters.")

    if "risk_score" in cust_txs.columns:
        top_risky = cust_txs.nlargest(3, "risk_score")
        for _, row in top_risky.iterrows():
            reasons = explain_tx(row, ds)
            for cat, lvl, msg in reasons:
                st.markdown(reason_card(cat, lvl, msg), unsafe_allow_html=True)

    st.markdown(section("Account Transactions"), unsafe_allow_html=True)
    display_cols = []
    if cfg["timestamp_col"] and cfg["timestamp_col"] in cust_txs.columns:
        display_cols.append(cfg["timestamp_col"])
    if amount_col in cust_txs.columns:
        display_cols.append(amount_col)
    for c in ["risk_lr", "risk_rf", "risk_xgb", "risk_score", "risk_level"]:
        if c in cust_txs.columns:
            display_cols.append(c)
    if target_col in cust_txs.columns:
        display_cols.append(target_col)
    if display_cols:
        show_df = cust_txs[display_cols].sort_values("risk_score", ascending=False) if "risk_score" in cust_txs else cust_txs[display_cols]
        st.dataframe(show_df.head(100), width="stretch", hide_index=True, height=350)

    c1, c2, c3 = st.columns([1, 2, 4])
    with c1:
        if st.button("Back"): go_to(2); st.rerun()
    with c2:
        if st.button("Analytics Dashboard", type="primary"): go_to(4); st.rerun()


# ═══════════════════════════════════════════════════════════════
# STEP 4 — ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    ds = st.session_state.dataset
    cfg = DATASETS[ds]
    st.markdown(section(f"Feature Engineering Analytics -- {ds}"), unsafe_allow_html=True)

    df = load_data(ds)
    models, scaler, features = load_models(ds)
    df_scored = score_transactions(df, models, scaler, features, ds)

    if ds == "AML Transactions":
        c1, c2 = st.columns(2)
        with c1:
            if "risk_level" in df_scored.columns:
                dist = df_scored["risk_level"].value_counts().reindex(["LOW","MEDIUM","HIGH","CRITICAL"]).fillna(0)
                fig = go.Figure(go.Bar(x=dist.index, y=dist.values,
                    marker_color=[RISK_COLORS[l] for l in dist.index],
                    text=dist.values, textposition="outside"))
                fig.update_layout(title="Transaction Risk Distribution", height=350, showlegend=False, **plotly_layout())
                st.plotly_chart(fig, width="stretch")
        with c2:
            if "Payment Format" in df_scored.columns:
                pf = df_scored.groupby("Payment Format")["risk_score"].mean().sort_values()
                fig = px.bar(pf, orientation="h", title="Avg Risk by Payment Format",
                    color=pf.values, color_continuous_scale="Reds")
                fig.update_layout(coloraxis_showscale=False, height=350, **plotly_layout())
                st.plotly_chart(fig, width="stretch")

        st.markdown(section("Temporal Behavior Tracking"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if "hour" in df_scored.columns:
                hr = df_scored.groupby("hour").agg(avg_risk=("risk_score","mean"), count=("risk_score","count")).reset_index()
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=hr["hour"], y=hr["count"], name="Tx Count", marker_color="#3B82B5", opacity=0.6), secondary_y=False)
                fig.add_trace(go.Scatter(x=hr["hour"], y=hr["avg_risk"], name="Avg Risk", line=dict(color="#9A3412", width=3), mode="lines+markers"), secondary_y=True)
                fig.update_layout(title="Hourly Activity Pattern", height=380, **plotly_layout())
                st.plotly_chart(fig, width="stretch")
        with c2:
            if "night_tx_ratio" in df_scored.columns:
                sub = df_scored.drop_duplicates("Account")[["night_tx_ratio"]].dropna()
                fig = px.histogram(sub, x="night_tx_ratio", nbins=25, title="Night Activity Distribution", color_discrete_sequence=["#1E5F9F"])
                fig.update_layout(height=380, **plotly_layout())
                st.plotly_chart(fig, width="stretch")

        st.markdown(section("Anomaly Indicators"), unsafe_allow_html=True)
        spike_cols = [c for c in ["amount_spike","tx_count_spike","receiver_spike"] if c in df_scored.columns]
        if spike_cols:
            data = []
            for col in spike_cols:
                label = col.replace("_"," ").title()
                w = df_scored[df_scored[col]==1]["risk_score"].mean()
                wo = df_scored[df_scored[col]==0]["risk_score"].mean()
                data.append({"Indicator": label, "With Spike": w, "Without": wo})
            melt = pd.DataFrame(data).melt(id_vars="Indicator", var_name="Condition", value_name="Avg Risk")
            fig = px.bar(melt, x="Indicator", y="Avg Risk", color="Condition", barmode="group",
                title="Risk by Anomaly Indicator", color_discrete_sequence=["#9A3412","#2D6A4F"])
            fig.update_layout(height=380, **plotly_layout())
            st.plotly_chart(fig, width="stretch")

        st.markdown(section("Cross-Bank and Currency Analysis"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if "is_cross_bank" in df_scored.columns:
                cb = df_scored.groupby("is_cross_bank")["risk_score"].mean().reset_index()
                cb["Type"] = cb["is_cross_bank"].map({0:"Same Bank",1:"Cross-Bank"})
                fig = px.bar(cb, x="Type", y="risk_score", color="Type", title="Same-Bank vs Cross-Bank",
                    color_discrete_sequence=["#2D6A4F","#9A3412"])
                fig.update_layout(showlegend=False, height=350, **plotly_layout())
                st.plotly_chart(fig, width="stretch")
        with c2:
            if "currency_mismatch" in df_scored.columns:
                cm = df_scored.groupby("currency_mismatch")["risk_score"].mean().reset_index()
                cm["Type"] = cm["currency_mismatch"].map({0:"Same Currency",1:"Mismatch"})
                fig = px.bar(cm, x="Type", y="risk_score", color="Type", title="Currency Match vs Mismatch",
                    color_discrete_sequence=["#2D6A4F","#9A3412"])
                fig.update_layout(showlegend=False, height=350, **plotly_layout())
                st.plotly_chart(fig, width="stretch")

        st.markdown(section("Monthly Tracking"), unsafe_allow_html=True)
        monthly_cols = [c for c in df_scored.columns if c.startswith("monthly_")]
        if monthly_cols:
            sel = st.selectbox("Select monthly feature:", monthly_cols)
            fig = px.histogram(df_scored, x=sel, nbins=30, title=f"Distribution: {sel}", color_discrete_sequence=["#1E5F9F"])
            fig.update_layout(height=350, **plotly_layout())
            st.plotly_chart(fig, width="stretch")

    elif ds == "PaySim":
        c1, c2 = st.columns(2)
        with c1:
            if "type" in df_scored.columns:
                tp = df_scored.groupby("type").agg(count=("risk_score","count"), avg_risk=("risk_score","mean")).reset_index().sort_values("avg_risk")
                fig = px.bar(tp, x="avg_risk", y="type", orientation="h", title="Avg Risk by Transaction Type",
                    color="avg_risk", color_continuous_scale="Reds")
                fig.update_layout(coloraxis_showscale=False, height=350, **plotly_layout())
                st.plotly_chart(fig, width="stretch")
        with c2:
            if "risk_level" in df_scored.columns:
                dist = df_scored["risk_level"].value_counts().reindex(["LOW","MEDIUM","HIGH","CRITICAL"]).fillna(0)
                fig = go.Figure(go.Bar(x=dist.index, y=dist.values,
                    marker_color=[RISK_COLORS[l] for l in dist.index], text=dist.values, textposition="outside"))
                fig.update_layout(title="Risk Distribution", height=350, showlegend=False, **plotly_layout())
                st.plotly_chart(fig, width="stretch")

        st.markdown(section("Feature Engineering Insights"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if "balance_diff_orig" in df_scored.columns:
                fig = px.histogram(df_scored, x="balance_diff_orig", nbins=30, title="Balance Difference (Sender)", color_discrete_sequence=["#1E5F9F"])
                fig.update_layout(height=350, **plotly_layout())
                st.plotly_chart(fig, width="stretch")
        with c2:
            if "amount_vs_balance" in df_scored.columns:
                fig = px.histogram(df_scored, x="amount_vs_balance", nbins=30, title="Amount vs Balance Ratio", color_discrete_sequence=["#9A3412"])
                fig.update_layout(height=350, **plotly_layout())
                st.plotly_chart(fig, width="stretch")

        if "dest_is_customer" in df_scored.columns:
            dest = df_scored.groupby("dest_is_customer")["risk_score"].mean().reset_index()
            dest["Type"] = dest["dest_is_customer"].map({0:"Merchant",1:"Customer"})
            fig = px.bar(dest, x="Type", y="risk_score", color="Type", title="Risk: Customer vs Merchant",
                color_discrete_sequence=["#2D6A4F","#9A3412"])
            fig.update_layout(showlegend=False, height=350, **plotly_layout())
            st.plotly_chart(fig, width="stretch")

    elif ds == "Credit Card":
        c1, c2 = st.columns(2)
        with c1:
            if "risk_level" in df_scored.columns:
                dist = df_scored["risk_level"].value_counts().reindex(["LOW","MEDIUM","HIGH","CRITICAL"]).fillna(0)
                fig = go.Figure(go.Bar(x=dist.index, y=dist.values,
                    marker_color=[RISK_COLORS[l] for l in dist.index], text=dist.values, textposition="outside"))
                fig.update_layout(title="Risk Distribution", height=350, showlegend=False, **plotly_layout())
                st.plotly_chart(fig, width="stretch")
        with c2:
            if "Amount" in df_scored.columns and "risk_score" in df_scored.columns:
                fig = px.scatter(df_scored.sample(min(2000,len(df_scored))),
                    x="Amount", y="risk_score", color="risk_level", color_discrete_map=RISK_COLORS,
                    title="Amount vs Risk Score", opacity=0.6)
                fig.update_layout(height=350, **plotly_layout())
                st.plotly_chart(fig, width="stretch")

        st.markdown(section("PCA Feature Analysis"), unsafe_allow_html=True)
        v_cols = [f"V{i}" for i in range(1, 29) if f"V{i}" in df_scored.columns]
        if v_cols and "risk_score" in df_scored.columns:
            corrs = df_scored[v_cols + ["risk_score"]].corr()["risk_score"].drop("risk_score")
            top10 = corrs.abs().nlargest(10)
            fig = px.bar(x=top10.index, y=corrs[top10.index], title="Top PCA Features vs Risk Score",
                color=corrs[top10.index].values, color_continuous_scale="RdBu_r")
            fig.update_layout(coloraxis_showscale=False, height=380, **plotly_layout())
            st.plotly_chart(fig, width="stretch")

    c1, c2, c3 = st.columns([1, 2, 4])
    with c1:
        if st.button("Back"): go_to(3); st.rerun()
    with c2:
        if st.button("Model Performance", type="primary"): go_to(5); st.rerun()


# ═══════════════════════════════════════════════════════════════
# STEP 5 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════
elif st.session_state.step == 5:
    ds = st.session_state.dataset
    st.markdown(section(f"Model Performance -- {ds}"), unsafe_allow_html=True)

    perf = MODEL_PERF.get(ds, {})
    if perf:
        perf_df = pd.DataFrame(perf).T.reset_index()
        perf_df.columns = ["Model"] + list(perf_df.columns[1:])

        st.dataframe(perf_df, width="stretch", hide_index=True)
        st.caption("Fraud-class metrics on the 20% stratified hold-out test split of the 30,000-transaction sample.")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(perf_df, x="Model", y=["Precision","Recall","F1"], barmode="group",
                title="Classification Metrics", color_discrete_sequence=["#0B2545","#1E5F9F","#3B82B5"])
            fig.update_layout(yaxis_range=[0,1.05], height=400, **plotly_layout())
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.bar(perf_df, x="Model", y=["ROC-AUC","PR-AUC"], barmode="group",
                title="AUC Metrics", color_discrete_sequence=["#0B2545","#B8860B"])
            fig.update_layout(yaxis_range=[0,1.05], height=400, **plotly_layout())
            st.plotly_chart(fig, width="stretch")

        models_loaded, scaler, feat_list = load_models(ds)
        if "rf" in models_loaded and hasattr(models_loaded["rf"], "feature_importances_"):
            fi = pd.DataFrame({"Feature": feat_list[:len(models_loaded["rf"].feature_importances_)],
                "Importance": models_loaded["rf"].feature_importances_}).sort_values("Importance", ascending=True)
            fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                title="Random Forest -- Feature Importance", color_discrete_sequence=["#1E5F9F"])
            fig.update_layout(height=max(350, len(fi)*20), **plotly_layout())
            st.plotly_chart(fig, width="stretch")

    st.markdown(section("Cross-Dataset Comparison"), unsafe_allow_html=True)
    all_rows = []
    for d, mp in MODEL_PERF.items():
        for m, metrics in mp.items():
            all_rows.append({"Dataset": d, "Model": m, **metrics})
    all_df = pd.DataFrame(all_rows)
    st.dataframe(all_df, width="stretch", hide_index=True)

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Back"): go_to(4); st.rerun()
    with c2:
        if st.button("Start New Session", type="primary"):
            go_to(1, dataset="", account="")
            st.rerun()
