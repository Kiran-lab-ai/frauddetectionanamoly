import pandas as pd
from src.feature_engineering import load_data, prepare_features, ML_FEATURES
from src.anomaly_detector import train_model
import joblib

history = pd.read_csv("metadata/historical_metrics.csv")

approved_dates = history[
    (history["approval_status"] == "APPROVED") &
    (history["learned_into_model"] == True)
]["batch_date"].astype(str).unique()

print("Approved batches eligible for training:", approved_dates)

training_df = pd.DataFrame()

for batch_date in approved_dates:
    batch = load_data("data/historical/bank_transactions_data_2.csv")
    batch["TransactionDate"] = pd.to_datetime(batch["TransactionDate"])

    batch = batch[
        batch["TransactionDate"].dt.date ==
        pd.to_datetime(batch_date).date()
    ]

    training_df = pd.concat([training_df, batch], ignore_index=True)

print("Training records:", len(training_df))

training_df = prepare_features(training_df)

X = training_df[ML_FEATURES].fillna(0)

print("Training features:", len(ML_FEATURES))

model = train_model(X)

joblib.dump(model, "models/model_pulse_v2.pkl")

print("Retraining completed")
print("Model saved: model_pulse_v2.pkl")