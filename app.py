import streamlit as st
from src.feature_engineering import load_data
from src.batch_detector import detect_batch
import joblib
import os
from datetime import datetime
import pandas as pd

def save_metrics(batch_date, batch, approval_type, approval_status, approved_by, learned):
    path = "metadata/historical_metrics.csv"
    record = pd.DataFrame([{
        "batch_date": batch_date,
        "processed_at": datetime.now().isoformat(),
        "total_transactions": len(batch),
        "anomaly_count": (batch["anomaly_flag"] == -1).sum(),
        "anomaly_rate": ((batch["anomaly_flag"] == -1).sum() / len(batch)) * 100,
        "approval_type": approval_type,
        "approval_status": approval_status,
        "approved_by": approved_by,
        "model_version": "PULSE-v1",
        "learned_into_model": learned
    }])
    if os.path.exists(path):
        existing = pd.read_csv(path)

        duplicate = existing[
            (existing["batch_date"] == batch_date) &
            (existing["approval_type"] == approval_type) &
            (existing["approval_status"] == approval_status)
        ]

    if not duplicate.empty:
        return
    record.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
st.set_page_config(page_title="Transaction Anomaly Detection", layout="wide")

st.title("🏦 Transaction Anomaly Detection Agent")

model = joblib.load("models/model_pulse_v1.pkl")
df = load_data("data/historical/bank_transactions_data_2.csv")

batch_date = st.selectbox(
    "Select Transaction Date",
    sorted(df["TransactionDate"].str[:10].unique(), reverse=True)
)

batch = detect_batch(model, df, batch_date)

total = len(batch)
anomalies = (batch["anomaly_flag"] == -1).sum()
anomaly_pct = (anomalies / total) * 100

if anomalies > 0:
    approval = "🚨 Business Approval Required"
else:
    approval = "✅ Auto Approved"

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", total)
col2.metric("Anomalies Detected", anomalies)
col3.metric("Anomaly Rate", f"{anomaly_pct:.2f}%")
col4.metric("Batch Status", approval)

st.subheader(f"Transaction Monitoring — {batch_date}")

display_df = batch[
    ["TransactionID", "TransactionAmount", "Location",
     "Channel", "anomaly_score", "anomaly_flag"]
].copy()

def highlight_anomaly(row):
    return [
        "background-color: #ffcccc; font-weight: bold"
        if row["anomaly_flag"] == -1 else ""
        for _ in row
    ]

st.dataframe(
    display_df.style.apply(highlight_anomaly, axis=1),
    use_container_width=True
)

st.divider()
st.subheader("🔍 Anomaly Investigation")

anomalies = batch[batch["anomaly_flag"] == -1]

if len(anomalies) > 0:
    transaction = anomalies.iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric("Transaction", transaction["TransactionID"])
    col2.metric("Transaction Amount", f"${transaction['TransactionAmount']:.2f}")
    col3.metric("Anomaly Score", f"{transaction['anomaly_score']:.2f}/100")

    st.warning("🚨 BUSINESS APPROVAL REQUIRED")

    st.write(
        "**Investigation reason:** "
        + transaction["anomaly_reason"]
    )

    history = pd.read_csv("metadata/historical_metrics.csv")

    previous = history[
        (history["batch_date"].astype(str) == str(batch_date)) &
        (history["approval_type"] == "MANUAL")
        ]

    if previous.empty:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Approve Batch"):
                save_metrics(batch_date, batch, "MANUAL", "APPROVED",
                            "BUSINESS_USER", True)
                st.success("Batch approved.")
                st.rerun()

        with col2:
            if st.button("❌ Reject Batch"):
                save_metrics(batch_date, batch, "MANUAL", "REJECTED",
                            "BUSINESS_USER", False)
                st.error("Batch rejected.")
                st.rerun()

    else:
        decision = previous.iloc[-1]["approval_status"]

        if decision == "APPROVED":
            st.success("✅ Batch already approved")
        else:
            st.error("❌ Batch already rejected.No model learning will occur.")
else:
    save_metrics(
        batch_date, batch, "AUTO", "APPROVED",
        "SYSTEM", True
    )

    st.success("✅ Batch automatically approved and recorded.")

st.divider()
st.subheader("📊 Historical Processing Metrics")

history_path = "metadata/historical_metrics.csv"

if os.path.exists(history_path):
    history = pd.read_csv(history_path)
    st.dataframe(history, use_container_width=True)
else:
    st.info("No historical batches processed yet.")