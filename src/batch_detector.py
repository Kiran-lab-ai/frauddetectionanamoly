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
    batch["raw_anomaly_score"] = -scores

    return batch