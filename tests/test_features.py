from src.feature_engineering import load_data, prepare_features

df = load_data("data/historical/bank_transactions_data_2.csv")
df = prepare_features(df)

print(df[
    [
        "TransactionAmount",
        "transaction_hour",
        "customer_avg_amount",
        "amount_vs_customer_avg",
        "location_is_new",
        "device_is_new",
        "channel_is_unusual",
        "amount_to_balance_ratio",
        "high_login_attempts",
        "duration_vs_customer_avg"
    ]
].head(10).to_string())