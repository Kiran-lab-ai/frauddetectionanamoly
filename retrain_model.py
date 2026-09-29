import os
import pandas as pd
import joblib

from src.feature_engineering import (
    load_data,
    prepare_features,
    ML_FEATURES
)

from src.anomaly_detector import train_model


# =========================================================
# Configuration
# =========================================================

HISTORICAL_DATA = (
    "data/historical/bank_transactions_data_2.csv"
)

VALIDATED_DATA = (
    "metadata/validated_transactions.csv"
)

MODEL_OUTPUT = (
    "models/model_pulse_v2.pkl"
)


# =========================================================
# Load Historical Baseline
# =========================================================

print("\n==========================================")
print("PULSE CONTROLLED RETRAINING")
print("==========================================")

historical_df = load_data(
    HISTORICAL_DATA
)

print(
    "Historical baseline records:",
    len(historical_df)
)


# =========================================================
# Load Customer-Verified Transactions
# =========================================================

if os.path.exists(
    VALIDATED_DATA
):

    validated_df = pd.read_csv(
        VALIDATED_DATA
    )

else:

    validated_df = pd.DataFrame()


# =========================================================
# Select Only Customer-Confirmed Genuine
# =========================================================

if not validated_df.empty:

    genuine_df = validated_df[
        validated_df[
            "eligible_for_learning"
        ] == True
    ].copy()

else:

    genuine_df = pd.DataFrame()


print(
    "Customer-confirmed genuine records:",
    len(genuine_df)
)


# =========================================================
# Controlled Learning Dataset
# =========================================================

training_df = historical_df.copy()


if not genuine_df.empty:

    # Keep only columns available in
    # the historical transaction dataset.

    genuine_transactions = genuine_df[
        historical_df.columns
    ].copy()

    # -----------------------------------------------------
    # Controlled feedback:
    # customer-confirmed genuine transactions receive
    # additional representation during retraining.
    # -----------------------------------------------------

    training_df = pd.concat(
        [
            training_df,
            genuine_transactions,
            genuine_transactions
        ],
        ignore_index=True
    )

    print(
        "Customer-confirmed transactions added "
        "to learning dataset:",
        len(genuine_transactions)
    )

else:

    print(
        "No customer-confirmed genuine transactions "
        "available for feedback learning."
    )


print(
    "Final training records:",
    len(training_df)
)


# =========================================================
# Prepare Features
# =========================================================

training_df = prepare_features(
    training_df
)


X = training_df[
    ML_FEATURES
].fillna(0)


print(
    "Training features:",
    len(ML_FEATURES)
)


# =========================================================
# Train PULSE Model
# =========================================================

model = train_model(
    X
)


# =========================================================
# Save New Model
# =========================================================

joblib.dump(
    model,
    MODEL_OUTPUT
)


print(
    "\n=========================================="
)

print(
    "PULSE retraining completed successfully."
)

print(
    "Model saved:",
    MODEL_OUTPUT
)

print(
    "==========================================\n"
)