import os
import httpx

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv


# =========================================================
# Load Environment Variables
# =========================================================

env_file = find_dotenv()

load_dotenv(
    env_file,
    override=True
)

api_key = os.getenv(
    "OPENAI_API_KEY"
)


print(
    "AI AGENT ENV:",
    env_file
)

print(
    "AI AGENT KEY PREFIX:",
    api_key[:8] if api_key else "NONE"
)

print(
    "AI AGENT KEY LENGTH:",
    len(api_key) if api_key else 0
)


# =========================================================
# TCS GenAI Client
# =========================================================

http_client = httpx.Client(
    verify=False
)

client = OpenAI(
    base_url="https://genailab.tcs.in",
    api_key=api_key,
    http_client=http_client
)


# =========================================================
# Static Fallback Investigation
# =========================================================

def static_investigation(transaction):

    risk_level = transaction.get(
        "risk_level",
        "Unknown"
    )

    score = transaction.get(
        "anomaly_score",
        0
    )

    amount = transaction.get(
        "TransactionAmount",
        0
    )

    location = transaction.get(
        "Location",
        "Unknown"
    )

    channel = transaction.get(
        "Channel",
        "Unknown"
    )

    login_attempts = transaction.get(
        "LoginAttempts",
        0
    )

    balance = transaction.get(
        "AccountBalance",
        0
    )

    # ---------------------------------------------
    # Why suspicious
    # ---------------------------------------------

    why = [
        (
            f"PULSE detected unusual transaction behaviour "
            f"with an anomaly score of {score:.2f}."
        ),
        (
            f"The transaction amount of ${amount:.2f} "
            f"was identified as unusual by the anomaly model."
        ),
        (
            f"The transaction occurred through the "
            f"{channel} channel from {location}."
        )
    ]

    # ---------------------------------------------
    # Risk signals
    # ---------------------------------------------

    risk = [
        (
            f"Risk classification: {risk_level}."
        ),
        (
            f"Login attempts recorded: {login_attempts}."
        ),
        (
            f"Current account balance: ${balance:.2f}."
        )
    ]

    if login_attempts >= 3:

        risk.append(
            "Multiple login attempts were detected."
        )

    # ---------------------------------------------
    # Recommended action
    # ---------------------------------------------

    action = [
        (
            "Contact the customer and verify whether "
            "the transaction was initiated by them."
        ),
        (
            "Review the customer's recent transaction "
            "activity for additional unusual behaviour."
        ),
        (
            "Do not treat the model anomaly as confirmed "
            "fraud without customer verification."
        )
    ]

    return {
        "why": why,
        "risk": risk,
        "action": action
    }


# =========================================================
# Parse AI Response
# =========================================================

def parse_ai_response(
    response_text,
    transaction
):

    try:

        why_start = response_text.find(
            "[WHY]"
        )

        risk_start = response_text.find(
            "[RISK]"
        )

        action_start = response_text.find(
            "[ACTION]"
        )

        # ---------------------------------------------
        # Validate required sections
        # ---------------------------------------------

        if (
            why_start == -1
            or risk_start == -1
            or action_start == -1
        ):

            print(
                "AI RESPONSE FORMAT INVALID - "
                "USING STATIC FALLBACK"
            )

            return static_investigation(
                transaction
            )

        # ---------------------------------------------
        # Extract sections
        # ---------------------------------------------

        why_text = response_text[
            why_start + len("[WHY]"):
            risk_start
        ].strip()

        risk_text = response_text[
            risk_start + len("[RISK]"):
            action_start
        ].strip()

        action_text = response_text[
            action_start + len("[ACTION]"):
        ].strip()

        # ---------------------------------------------
        # Convert lines to bullet lists
        # ---------------------------------------------

        def clean_bullets(text):

            lines = text.splitlines()

            bullets = []

            for line in lines:

                line = line.strip()

                if not line:
                    continue

                # Remove common bullet characters
                line = line.lstrip(
                    "-•* "
                )

                # Remove accidental numbering
                if (
                    len(line) > 2
                    and line[0].isdigit()
                    and line[1] in [".", ")"]
                ):
                    line = line[2:].strip()

                if line:
                    bullets.append(line)

            return bullets

        why = clean_bullets(
            why_text
        )

        risk = clean_bullets(
            risk_text
        )

        action = clean_bullets(
            action_text
        )

        # ---------------------------------------------
        # Validate extracted content
        # ---------------------------------------------

        if not why or not risk or not action:

            print(
                "AI RESPONSE SECTIONS EMPTY - "
                "USING STATIC FALLBACK"
            )

            return static_investigation(
                transaction
            )

        return {
            "why": why,
            "risk": risk,
            "action": action
        }

    except Exception as e:

        print(
            "AI RESPONSE PARSING ERROR:",
            e
        )

        return static_investigation(
            transaction
        )


# =========================================================
# AI Investigation
# =========================================================

def investigate_transaction(
    transaction
):

    ai_prompt = f"""
You are PULSE, a banking transaction investigation
assistant.

Analyze the following transaction that has been flagged
as anomalous by the PULSE anomaly detection model.

Transaction ID: {transaction['TransactionID']}
Amount: {transaction['TransactionAmount']}
Account: {transaction['AccountID']}
Location: {transaction['Location']}
Channel: {transaction['Channel']}
Device: {transaction['DeviceID']}
Login Attempts: {transaction['LoginAttempts']}
Account Balance: {transaction['AccountBalance']}
Anomaly Score: {transaction['anomaly_score']}
Risk Level: {transaction['risk_level']}

Your response MUST follow this exact structure:

[WHY]
- Provide 2 to 3 concise bullet points explaining why
  this transaction may be unusual.

[RISK]
- Provide 2 to 3 concise bullet points describing the
  key risk signals.

[ACTION]
- Provide 2 to 3 concise bullet points describing the
  recommended investigation action for Customer Support.

IMPORTANT:

- Use ONLY the three markers:
  [WHY]
  [RISK]
  [ACTION]

- Do not create any other headings.
- Do not use #, ## or ###.
- Do not create a title.
- Do not repeat the transaction ID as a heading.
- Use bullet points only inside each section.
- Keep the response concise.
- Base statements only on the transaction information provided.
- Do not invent customer history or behaviour.
- Do not claim the transaction is definitely fraud.
- An anomaly is not confirmation of fraud.
- Customer Support should verify the transaction with
  the customer.
"""

    try:

        response = client.responses.create(
            model="azure/genailab-maas-gpt-4o-mini",
            input=ai_prompt
        )

        response_text = (
            response.output_text
            if response.output_text
            else ""
        )

        # ---------------------------------------------
        # Empty response
        # ---------------------------------------------

        if not response_text.strip():

            print(
                "AI RESPONSE EMPTY - "
                "USING STATIC FALLBACK"
            )

            return static_investigation(
                transaction
            )

        print(
            "PULSE AI RESPONSE GENERATED"
        )

        return parse_ai_response(
            response_text,
            transaction
        )

    except Exception as e:

        print(
            "OPENAI ERROR:",
            e
        )

        print(
            "USING STATIC INVESTIGATION FALLBACK"
        )

        return static_investigation(
            transaction
        )


# =========================================================
# Standalone Test
# =========================================================

if __name__ == "__main__":

    try:

        response = client.responses.create(
            model="azure/genailab-maas-gpt-4o-mini",
            input=(
                "Reply with exactly: "
                "PULSE TEST SUCCESS"
            )
        )

        print(
            "\n===== PULSE TEST ====="
        )

        print(
            response.output_text
        )

    except Exception as e:

        print(
            "\n===== PULSE TEST FAILED ====="
        )

        print(
            e
        )