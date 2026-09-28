import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def prepare_features(df):
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])

    df["transaction_hour"] = df["TransactionDate"].dt.hour
    df["transaction_day"] = df["TransactionDate"].dt.dayofweek
    df["is_weekend"] = (df["transaction_day"] >= 5).astype(int)

    customer_avg = df.groupby("AccountID")["TransactionAmount"].transform("mean")
    customer_std = df.groupby("AccountID")["TransactionAmount"].transform("std")

    df["customer_avg_amount"] = customer_avg
    df["customer_std_amount"] = customer_std.fillna(0)

    df["amount_vs_customer_avg"] = (
        df["TransactionAmount"] / df["customer_avg_amount"]
    )

    customer_location_count = df.groupby(
        ["AccountID", "Location"]
    )["TransactionID"].transform("count")

    customer_transaction_count = df.groupby(
        "AccountID"
    )["TransactionID"].transform("count")

    df["location_is_new"] = (
        customer_location_count == 1
    ).astype(int)


    customer_device_count = df.groupby(
        ["AccountID", "DeviceID"]
    )["TransactionID"].transform("count")

    df["device_is_new"] = (
        customer_device_count == 1
    ).astype(int)


    customer_channel_count = df.groupby(
        ["AccountID", "Channel"]
    )["TransactionID"].transform("count")

    df["channel_is_unusual"] = (
        customer_channel_count == 1
    ).astype(int)


    df["amount_to_balance_ratio"] = (
        df["TransactionAmount"] / df["AccountBalance"].replace(0, 1)
    )

    df["high_login_attempts"] = (
        df["LoginAttempts"] >= 3
    ).astype(int)


    customer_avg_duration = df.groupby(
        "AccountID"
    )["TransactionDuration"].transform("mean")

    df["duration_vs_customer_avg"] = (
        df["TransactionDuration"] / customer_avg_duration
    )

    return df

ML_FEATURES = [
    "TransactionAmount",
    "customer_avg_amount",
    "customer_std_amount",
    "amount_vs_customer_avg",
    "transaction_hour",
    "transaction_day",
    "is_weekend",
    "location_is_new",
    "device_is_new",
    "channel_is_unusual",
    "LoginAttempts",
    "TransactionDuration",
    "AccountBalance",
    "amount_to_balance_ratio",
    "high_login_attempts",
    "duration_vs_customer_avg"
]