import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request

app = Flask(__name__)
# Configure logging
logging.basicConfig(filename="server.log", level=logging.DEBUG)

FEATURES = [
    "day_sin",
    "day_cos",
    "high_diff",
    "low_diff",
    "open",
    "close",
    "volume",
]
EMA_PERIOD = 5
SEQ_LENGTH = 15


# Endpoint for handling POST requests
@app.route("/predict", methods=["POST"])
def handle_post_request():
    """
    Require:
    {
        datetime: [seconds]
        high: [float]
        low: [float]
        open: [float]
        close: [float]
        volume: [int]
        balance: float
        position: int
        entryPrice: float
    }
    Sequence Length = 16 (15 + 1 for pct change)
    """
    logging.debug(f"Request received: {request}")
    content = request.json
    logging.debug(f"Content received: {content}")

    columns = ["high", "low", "open", "close", "volume"]
    data = pd.DataFrame({c: content[c] for c in columns})

    day = 24 * 60 * 60
    current_time = datetime.now()
    twenty_minutes_ago = current_time - timedelta(minutes=20)
    twenty_minutes_ago_seconds = int(twenty_minutes_ago.timestamp())
    timestamp_s = np.array([twenty_minutes_ago_seconds + i for i in range(0, 1200, 60)])

    data["day_sin"] = np.sin(timestamp_s * (2 * np.pi / day))
    data["day_cos"] = np.cos(timestamp_s * (2 * np.pi / day))

    for col in ["high", "low"]:
        data[f"{col}_diff"] = (data[col] - data["close"]).abs()

    for col in ["open", "close"]:
        data[f"{col}_pct"] = data[col].pct_change()

    # Make a request to the model
    observation = np.concatenate(
        [
            [content["balance"]],
            [content["position"]],
            [content["entryPrice"]],
            *[data[feature][-SEQ_LENGTH:].values for feature in FEATURES],
        ],
        dtype=np.float32,
    )

    input_data = {
        "0/observation": observation.tolist(),
        "0/step_type": 1,  # StepType.Mid
        "0/discount": 1.0,  # Dummy
        "0/reward": 0,  # Dummy
    }

    model_response = requests.post(
        "http://tensorflow-serve:8501/v1/models/model:predict",
        json={"signature_name": "action", "instances": [input_data]},
    )
    logging.debug(f"Model response: {model_response}")
    logging.debug(f"Model content: {model_response.json()}")

    prediction: float = model_response.json()["predictions"][0]
    logging.debug(f"Model prediction: {prediction}")

    response = {"prediction": prediction}

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
