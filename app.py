import streamlit as st

from src.feature_engineering import load_data
from src.batch_detector import detect_batch
from src.ai_agent import investigate_transaction

import joblib
import os

from datetime import datetime

import pandas as pd


# =========================================================
# Streamlit Configuration
# =========================================================

st.set_page_config(
    page_title="Transaction Anomaly Detection",
    layout="wide"
)

st.title(
    "🏦 Transaction Anomaly Detection Agent"
)


# =========================================================
# Load Model and Data
# =========================================================

model = joblib.load(
    "models/model_pulse_v1.pkl"
)

df = load_data(
    "data/historical/bank_transactions_data_2.csv"
)


# =========================================================
# Select Transaction Date
# =========================================================

batch_date = st.selectbox(
    "Select Transaction Date",
    sorted(
        df["TransactionDate"]
        .str[:10]
        .unique(),
        reverse=True
    )
)


# =========================================================
# Detect Anomalies
# =========================================================

batch = detect_batch(
    model,
    df,
    batch_date
)


# =========================================================
# KPI Metrics
# =========================================================

total = len(batch)

anomalies = (
    batch["anomaly_flag"] == -1
).sum()

anomaly_pct = (
    anomalies / total
) * 100


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
    "Support Required"
    if anomalies > 0
    else "No Anomalies"
)


if anomalies > 0:

    col4.caption(
        "📞 Customer verification required"
    )

else:

    col4.caption(
        "✅ Batch is clear"
    )


# =========================================================
# Transaction Monitoring
# =========================================================

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
        (
            "background-color: #ffcccc; "
            "font-weight: bold"
        )
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


# =========================================================
# Batch EDA
# =========================================================

st.divider()

st.subheader(
    "📊 Batch EDA"
)

st.write(
    "Explore transaction behaviour in the incoming batch "
    "before investigating individual anomalies."
)


eda_col1, eda_col2 = st.columns(2)


# =========================================================
# Transaction Amount Distribution
# =========================================================

with eda_col1:

    st.markdown(
        "### 💰 Transaction Amount Distribution"
    )


    amount_bins = pd.cut(
        batch["TransactionAmount"],
        bins=10
    )


    amount_distribution = (
        amount_bins
        .value_counts()
        .sort_index()
    )


    amount_distribution.index = [
        (
            f"${interval.left:.0f}"
            f"–"
            f"${interval.right:.0f}"
        )
        for interval
        in amount_distribution.index
    ]


    st.bar_chart(
        amount_distribution,
        use_container_width=True
    )


# =========================================================
# Transaction Channel Distribution
# =========================================================

with eda_col2:

    st.markdown(
        "### 🏦 Transactions by Channel"
    )


    channel_distribution = (
        batch["Channel"]
        .value_counts()
    )


    st.bar_chart(
        channel_distribution,
        use_container_width=True
    )


# =========================================================
# Anomaly Investigation
# =========================================================

st.divider()

st.subheader(
    "🔍 Anomaly Investigation"
)


anomalies_df = batch[
    batch["anomaly_flag"] == -1
].copy()


if len(anomalies_df) > 0:

    st.write(
        "Select an anomaly transaction to generate "
        "its investigation report."
    )


    # =====================================================
    # Anomaly Selection Table
    # =====================================================

    anomaly_list = anomalies_df[
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
        selection_mode="single-row",
        key="anomaly_selection"
    )


    selected_rows = (
        selected_anomaly.selection.rows
    )


    # =====================================================
    # Selected Transaction
    # =====================================================

    selected_transaction = None


    if selected_rows:

        selected_transaction = (
            anomalies_df.iloc[
                selected_rows[0]
            ]
        )


        selected_transaction_id = (
            selected_transaction[
                "TransactionID"
            ]
        )


        previous_transaction_id = (
            st.session_state.get(
                "selected_transaction_id"
            )
        )


        if (
            previous_transaction_id is not None
            and
            previous_transaction_id
            != selected_transaction_id
        ):

            st.session_state[
                "view_report"
            ] = False


        st.session_state[
            "selected_transaction_id"
        ] = selected_transaction_id


    else:

        st.session_state[
            "view_report"
        ] = False

        selected_transaction = None


    # =====================================================
    # View Report Button
    # =====================================================

    if selected_transaction is not None:

        if st.button(
            "👁️ View Report",
            key="view_report_button"
        ):

            st.session_state[
                "view_report"
            ] = True


            st.session_state[
                "report_transaction_id"
            ] = selected_transaction[
                "TransactionID"
            ]


            st.rerun()


    # =====================================================
    # Display Investigation Report
    # =====================================================

    if st.session_state.get(
        "view_report",
        False
    ):

        report_transaction_id = (
            st.session_state.get(
                "report_transaction_id"
            )
        )


        matching_transactions = (
            anomalies_df[
                anomalies_df[
                    "TransactionID"
                ]
                ==
                report_transaction_id
            ]
        )


        if not matching_transactions.empty:

            transaction = (
                matching_transactions.iloc[0]
            )


            # =================================================
            # Transaction Summary
            # =================================================

            st.divider()

            summary_col1, summary_col2, summary_col3 = (
                st.columns(3)
            )


            summary_col1.metric(
                "Transaction",
                transaction[
                    "TransactionID"
                ]
            )


            summary_col2.metric(
                "Transaction Amount",
                f"${transaction['TransactionAmount']:.2f}"
            )


            summary_col3.metric(
                "Risk Level",
                transaction[
                    "risk_level"
                ]
            )


            # =================================================
            # Investigation Report
            # =================================================

            st.subheader(
                "🤖 AI Investigation Report"
            )


            report_col1, report_col2 = (
                st.columns(2)
            )


            # =================================================
            # LEFT — ANOMALY ALERT
            # =================================================

            with report_col1:

                st.markdown(
                    "### 🚨 ANOMALY ALERT"
                )


                st.write(
                    f"**Transaction ID:** "
                    f"{transaction['TransactionID']}"
                )


                st.write(
                    f"**Account ID:** "
                    f"{transaction['AccountID']}"
                )


                st.write(
                    f"**Amount:** "
                    f"${transaction['TransactionAmount']:.2f}"
                )


                st.write(
                    f"**Location:** "
                    f"{transaction['Location']}"
                )


                st.write(
                    f"**Channel:** "
                    f"{transaction['Channel']}"
                )


                st.write(
                    f"**Device:** "
                    f"{transaction['DeviceID']}"
                )


                st.write(
                    f"**Login Attempts:** "
                    f"{transaction['LoginAttempts']}"
                )


                st.write(
                    f"**Account Balance:** "
                    f"${transaction['AccountBalance']:.2f}"
                )


            # =================================================
            # RIGHT — AI ANALYSIS
            # =================================================

            with report_col2:

                st.markdown(
                    "### 🔎 Why This Transaction May Be Suspicious"
                )


                with st.spinner(
                    "🤖 Generating AI investigation report..."
                ):

                    investigation = (
                        investigate_transaction(
                            transaction
                        )
                    )


                # =================================================
                # WHY
                # =================================================

                for item in investigation["why"]:

                    st.markdown(
                        f"- {item}"
                    )


                # =================================================
                # KEY RISK SIGNALS
                # =================================================

                st.markdown(
                    "### ⚠️ Key Risk Signals"
                )


                for item in investigation["risk"]:

                    st.markdown(
                        f"- {item}"
                    )


                # =================================================
                # RECOMMENDED ACTION
                # =================================================

                st.markdown(
                    "### 📞 Recommended Investigation "
                    "Action for Customer Support"
                )


                for item in investigation["action"]:

                    st.markdown(
                        f"- {item}"
                    )


            # =================================================
            # Customer Support Verification
            # =================================================

            st.divider()

            st.info(
                "📞 CUSTOMER SUPPORT VERIFICATION REQUIRED"
            )


            verified_path = (
                "metadata/validated_transactions.csv"
            )


            already_verified = False


            if os.path.exists(
                verified_path
            ):

                verified_data = pd.read_csv(
                    verified_path
                )


                already_verified = (
                    transaction[
                        "TransactionID"
                    ]
                    in verified_data[
                        "TransactionID"
                    ].values
                )


            # =================================================
            # Verification Dropdown
            # =================================================

            if already_verified:

                previous_status = (
                    verified_data.loc[
                        verified_data[
                            "TransactionID"
                        ]
                        ==
                        transaction[
                            "TransactionID"
                        ],
                        "customer_verification"
                    ].iloc[-1]
                )


                st.selectbox(
                    "Customer Verification",
                    [previous_status],
                    disabled=True,
                    key=(
                        f"verified_"
                        f"{transaction['TransactionID']}"
                    )
                )


                st.success(
                    "✅ Customer verification completed: "
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
                    key=(
                        f"verification_"
                        f"{transaction['TransactionID']}"
                    )
                )


            # =================================================
            # Get Verification Status
            # =================================================

            if already_verified:

                verification_status = (
                    previous_status
                )

            else:

                verification_status = (
                    st.session_state[
                        f"verification_"
                        f"{transaction['TransactionID']}"
                    ]
                )


            # =================================================
            # Verification Message
            # =================================================

            if (
                verification_status
                ==
                "Customer Confirmed - Genuine"
            ):

                st.success(
                    "✅ Transaction verified by "
                    "customer as genuine."
                )


            elif (
                verification_status
                ==
                "Customer Declined - Not Genuine"
            ):

                st.error(
                    "❌ Transaction not verified "
                    "by customer."
                )


            # =================================================
            # Record Customer Verification
            # =================================================

            if not already_verified:

                if st.button(
                    "📞 Record Customer Verification",
                    key=(
                        f"record_"
                        f"{transaction['TransactionID']}"
                    )
                ):

                    if (
                        verification_status
                        ==
                        "Pending Customer Verification"
                    ):

                        st.warning(
                            "Please select the customer's "
                            "verification outcome first."
                        )


                    else:

                        verified_row = (
                            transaction
                            .to_frame()
                            .T
                            .copy()
                        )


                        verified_row[
                            "customer_verification"
                        ] = (
                            verification_status
                        )


                        verified_row[
                            "verified_at"
                        ] = (
                            datetime.now()
                            .isoformat()
                        )


                        verified_row[
                            "eligible_for_learning"
                        ] = (
                            verification_status
                            ==
                            "Customer Confirmed - Genuine"
                        )


                        # =====================================
                        # Save Verification
                        # =====================================

                        if os.path.exists(
                            verified_path
                        ):

                            existing = (
                                pd.read_csv(
                                    verified_path
                                )
                            )


                            if (
                                transaction[
                                    "TransactionID"
                                ]
                                not in
                                existing[
                                    "TransactionID"
                                ].values
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
                            "✅ Customer verification "
                            "recorded: "
                            f"{verification_status}"
                        )


                        st.rerun()


            # =================================================
            # Investigation Reason
            # =================================================

            st.write(
                "**Investigation reason:** "
                + transaction[
                    "anomaly_reason"
                ]
            )


else:

    st.success(
        "✅ No anomalies detected for this batch."
    )


# =========================================================
# Customer-Verified Transactions
# =========================================================

st.divider()

st.subheader(
    "✅ Customer-Verified Transactions"
)


validated_path = (
    "metadata/validated_transactions.csv"
)


if os.path.exists(
    validated_path
):

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