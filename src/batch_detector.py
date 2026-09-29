import pandas as pd
from src.feature_engineering import prepare_features, ML_FEATURES
from src.anomaly_detector import score_transactions

def detect_batch(model, df, batch_date):
    df = prepare_features(df)

    batch = df[
        df["TransactionDate"].dt.date == pd.to_datetime(batch_date).date()
    ].copy()

    X = batch[ML_FEATURES].fillna(0)

    predictions, scores = score_transactions(model, X)

    batch["anomaly_flag"] = predictions
    raw_scores = -scores

    batch["anomaly_score"] = (
    (raw_scores - raw_scores.min())
    / (raw_scores.max() - raw_scores.min())
    * 100
    if raw_scores.max() != raw_scores.min()
    else 0
)


    def classify_risk(score):
        if score >= 90:
            return "Critical"
        elif score >= 75:
            return "High"
        elif score >= 50:
            return "Medium"
        else:
            return "Low"

    batch["risk_level"] = batch["anomaly_score"].apply(classify_risk)
    
    batch["anomaly_reason"] = "Normal"
    batch.loc[batch["anomaly_flag"] == -1, "anomaly_reason"] = (
        "Unusual transaction behaviour detected by the model"
    )

    return batch