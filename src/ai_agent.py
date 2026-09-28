import os
from click import prompt
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def investigate_transaction(transaction):
    ai_prompt = f"""
You are a banking transaction investigation assistant.

Analyze this transaction flagged as anomalous by a model:

Transaction ID: {transaction['TransactionID']}
Amount: {transaction['TransactionAmount']}
Account: {transaction['AccountID']}
Location: {transaction['Location']}
Channel: {transaction['Channel']}
Login Attempts: {transaction['LoginAttempts']}
Account Balance: {transaction['AccountBalance']}
Anomaly Score: {transaction['anomaly_score']}

Provide:
1. Why this transaction may be suspicious
2. Key risk signals
3. Recommended investigation action

Do not claim that the transaction is definitely fraud.
"""
    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=ai_prompt
        )
        return response.output_text

    except Exception:
        return (
            f"Risk assessment: Anomalous transaction detected.\n"
            f"Transaction amount: {transaction['TransactionAmount']}\n"
            f"Anomaly score: {transaction['anomaly_score']:.2f}/100\n"
            f"Investigation action: Business review required."
        )