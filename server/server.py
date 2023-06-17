import requests
from flask import Flask, request, jsonify
import logging
import pandas as pd

#! Ensure to keep the same order like in training
HEADERS = [
    "progress",
    "open",
    "high",
    "low",
    "close",
    "volume",
]
STATE_VALS = [
    "contracts",
    "entryPrice",
    "balance",
]

PPC = 12.50
LIMIT = 75
THRESHOLD = 0.2

app = Flask(__name__)
# Configure logging
logging.basicConfig(filename="server.log", level=logging.DEBUG)


# Endpoint for handling GET requests
@app.route("/predict", methods=["POST"])
def handle_get_request():
    logging.debug(f"Request received: {request}")

    # Read data in order
    data = pd.DataFrame({header: request.json[header] for header in HEADERS})
    for sv in STATE_VALS:
        data[sv] = [request.json[sv]] * len(request.json[HEADERS[0]])
    logging.debug(f"Data read: {str(data)}")

    # Make a request to the model
    model_response = requests.post(
        "http://tensorflow-serve:8501/v1/models/prototype-V1:predict",
        json={"instances": [data.to_numpy().tolist()]},
    )
    logging.debug(f"Model Response: {model_response.json()}")

    predicted_result: float = model_response.json()

    return jsonify({"result": predicted_result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
