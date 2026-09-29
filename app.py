import streamlit as st
from src.feature_engineering import load_data
from src.batch_detector import detect_batch
import joblib
import os
from datetime import datetime
import pandas as pd


def save_metrics(
    batch_date,
    batch,
    approval_type,
    approval_status,
    approved_by,
    learned
):
    path = "metadata/historical_metrics.csv"

    record = pd.DataFrame([{
        "batch_date": batch_date,
        "processed_at": datetime.now().isoformat(),
        "total_transactions": len(batch),
        "anomaly_count": (batch["anomaly_flag"] == -1).sum(),
        "anomaly_rate": (
            (batch["anomaly_flag"] == -1).sum() / len(batch)
        ) * 100,
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

    record.to_csv(
        path,
        mode="a",
        header=not os.path.exists(path),
        index=False
    )


# ---------------------------------------------------------
# Streamlit Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Transaction Anomaly Detection",
    layout="wide"
)

st.title("🏦 Transaction Anomaly Detection Agent")


# ---------------------------------------------------------
# Load Model and Data
# ---------------------------------------------------------

model = joblib.load("models/model_pulse_v1.pkl")

df = load_data(
    "data/historical/bank_transactions_data_2.csv"
)


# ---------------------------------------------------------
# Select Transaction Date
# ---------------------------------------------------------

batch_date = st.selectbox(
    "Select Transaction Date",
    sorted(
        df["TransactionDate"].str[:10].unique(),
        reverse=True
    )
)


# ---------------------------------------------------------
# Detect Anomalies
# ---------------------------------------------------------

batch = detect_batch(
    model,
    df,
    batch_date
)


# ---------------------------------------------------------
# KPI Metrics
# ---------------------------------------------------------

total = len(batch)

anomalies = (
    batch["anomaly_flag"] == -1
).sum()

anomaly_pct = (
    anomalies / total
) * 100


if anomalies > 0:
    approval = "📞 Customer Support Verification Required"
else:
    approval = "✅ No Anomalies"


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Transactions",
    total
)

col2.metric(
    "Anomalies Detected",
    anomalies
)

col3.metric(
    "Anomaly Rate",
    f"{anomaly_pct:.2f}%"
)

col4.metric(
    "Batch Status",
    approval
)


# ---------------------------------------------------------
# Transaction Monitoring
# ---------------------------------------------------------

st.subheader(
    f"Transaction Monitoring — {batch_date}"
)

display_df = batch[
    [
        "TransactionID",
        "TransactionAmount",
        "Location",
        "Channel",
        "anomaly_score",
        "risk_level",
        "anomaly_flag"
    ]
].copy()


def highlight_anomaly(row):
    return [
        "background-color: #ffcccc; font-weight: bold"
        if row["anomaly_flag"] == -1
        else ""
        for _ in row
    ]


st.dataframe(
    display_df.style.apply(
        highlight_anomaly,
        axis=1
    ),
    use_container_width=True
)


# ---------------------------------------------------------
# Anomaly Investigation
# ---------------------------------------------------------

st.divider()

st.subheader(
    "🔍 Anomaly Investigation"
)

anomalies = batch[
    batch["anomaly_flag"] == -1
]
st.write("Select an anomaly transaction for customer verification")

anomaly_list = anomalies[
    [
        "TransactionID",
        "TransactionAmount",
        "Location",
        "Channel",
        "anomaly_score",
        "risk_level"
    ]
].copy()

selected_anomaly = st.dataframe(
    anomaly_list,
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row"
)

if len(anomalies) > 0:

    # -----------------------------------------------------
    # Selected Anomaly
    # -----------------------------------------------------

    if len(anomalies) > 0:

        selected_rows = selected_anomaly.selection.rows

    if selected_rows:
        transaction = anomalies.iloc[selected_rows[0]]
    else:
        transaction = anomalies.iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Transaction",
        transaction["TransactionID"]
    )

    col2.metric(
        "Transaction Amount",
        f"${transaction['TransactionAmount']:.2f}"
    )

    col3.metric(
        "Risk Level",
        transaction["risk_level"]
    )


    # -----------------------------------------------------
    # Customer Support Verification
    # -----------------------------------------------------

    st.info(
        "📞 CUSTOMER SUPPORT VERIFICATION REQUIRED"
    )

    verified_path = (
        "metadata/validated_transactions.csv"
    )

    already_verified = False


    if os.path.exists(verified_path):

        verified_data = pd.read_csv(
            verified_path
        )

        already_verified = (
            transaction["TransactionID"]
            in verified_data["TransactionID"].values
        )


    # -----------------------------------------------------
    # Verification Dropdown
    # -----------------------------------------------------

    if already_verified:

        previous_status = verified_data.loc[
            verified_data["TransactionID"]
            == transaction["TransactionID"],
            "customer_verification"
        ].iloc[-1]

        st.selectbox(
            "Customer Verification",
            [previous_status],
            disabled=True
        )

        st.success(
            f"✅ Customer verification completed: "
            f"{previous_status}"
        )

    else:

        st.selectbox(
            "Customer Verification",
            [
                "Pending Customer Verification",
                "Customer Confirmed - Genuine",
                "Customer Declined - Not Genuine"
            ],
            key=f"verification_{transaction['TransactionID']}"
        )


    # -----------------------------------------------------
    # Get Current Verification Status
    # -----------------------------------------------------

    if already_verified:

        verification_status = previous_status

    else:

        verification_status = st.session_state[
            f"verification_{transaction['TransactionID']}"
        ]


    # -----------------------------------------------------
    # Display Verification Status
    # -----------------------------------------------------

    if verification_status == (
        "Customer Confirmed - Genuine"
    ):

        st.success(
            "✅ Transaction verified by customer as genuine."
        )

    elif verification_status == (
        "Customer Declined - Not Genuine"
    ):

        st.error(
            "❌ Transaction not verified by customer."
        )


    # -----------------------------------------------------
    # Record Customer Verification
    # -----------------------------------------------------

    if not already_verified:

        if st.button(
            "📞 Record Customer Verification"
        ):

            if verification_status == (
                "Pending Customer Verification"
            ):

                st.warning(
                    "Please select the customer's "
                    "verification outcome first."
                )

            else:

                verified_row = (
                    transaction.to_frame()
                    .T.copy()
                )

                verified_row[
                    "customer_verification"
                ] = verification_status

                verified_row[
                    "verified_at"
                ] = datetime.now().isoformat()

                verified_row[
                    "eligible_for_learning"
                ] = (
                    verification_status
                    == "Customer Confirmed - Genuine"
                )


                # -----------------------------------------
                # Save Verification
                # -----------------------------------------

                if os.path.exists(
                    verified_path
                ):

                    existing = pd.read_csv(
                        verified_path
                    )

                    if transaction["TransactionID"] not in (
                        existing["TransactionID"].values
                    ):

                        verified_row.to_csv(
                            verified_path,
                            mode="a",
                            header=False,
                            index=False
                        )

                else:

                    verified_row.to_csv(
                        verified_path,
                        index=False
                    )


                st.success(
                    "✅ Customer verification recorded: "
                    f"{verification_status}"
                )
                st.rerun()

    # -----------------------------------------------------
    # Investigation Reason
    # -----------------------------------------------------

    st.write(
        "**Investigation reason:** "
        + transaction["anomaly_reason"]
    )


else:

    st.success(
        "✅ No anomalies detected for this batch."
    )


# ---------------------------------------------------------
# Customer-Verified Transactions
# ---------------------------------------------------------

st.divider()

st.subheader(
    "✅ Customer-Verified Transactions"
)

validated_path = "metadata/validated_transactions.csv"

if os.path.exists(validated_path):

    validated_data = pd.read_csv(
        validated_path
    )

    st.dataframe(
        validated_data[
            [
                "TransactionID",
                "TransactionAmount",
                "risk_level",
                "customer_verification",
                "eligible_for_learning"
            ]
        ],
        use_container_width=True
    )

else:

    st.info(
        "No customer-verified transactions yet."
    )