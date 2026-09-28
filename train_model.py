import pandas as pd
from src.feature_engineering import load_data, prepare_features, ML_FEATURES
from src.anomaly_detector import train_model
import joblib
from src.ai_agent import investigate_transaction

df = load_data("data/historical/bank_transactions_data_2.csv")
df = prepare_features(df)

X = df[ML_FEATURES].fillna(0)

model = train_model(X)

joblib.dump(model, "models/model_pulse_v1.pkl")
print("Model saved successfully")

print("Model trained successfully")
print("Records:", len(X))
print("Features:", len(ML_FEATURES))


from src.anomaly_detector import score_transactions

predictions, scores = score_transactions(model, X)

df["anomaly_flag"] = predictions
raw_scores = -scores

df["anomaly_score"] = (
    (raw_scores - raw_scores.min())
    / (raw_scores.max() - raw_scores.min())
    * 100
)

print(df[["TransactionID", "TransactionAmount", "anomaly_flag", "anomaly_score"]]
      .sort_values("anomaly_score", ascending=False)
      .head(10)
      .to_string(index=False))


from src.batch_detector import detect_batch

batch = detect_batch(
    model,
    df,
    "2023-12-25"
)

print("Batch records:", len(batch))
print("Anomalies:", (batch["anomaly_flag"] == -1).sum())
print(batch[
    ["TransactionID", "anomaly_flag",
     "amount_vs_customer_avg", "location_is_new",
     "anomaly_reason"]
].to_string(index=False))

anomaly_transactions = batch[batch["anomaly_flag"] == -1]

if len(anomaly_transactions) > 0:
    print("\n🚨 BUSINESS APPROVAL REQUIRED")
else:
    print("\n✅ AUTO APPROVED")

for _, transaction in anomaly_transactions.iterrows():
    print("\n===== AI INVESTIGATION =====")
    print(investigate_transaction(transaction))