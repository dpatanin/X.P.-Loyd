import requests
from flask import Flask, request, jsonify
import logging
import pandas as pd
import yaml
from yaml.loader import FullLoader
from lib.state import State
from lib.constants import BALANCE, ENTRY_PRICE, CONTRACTS

with open("config.yaml") as f:
    config = yaml.load(f, Loader=FullLoader)

app = Flask(__name__)
# Configure logging
logging.basicConfig(filename="server.log", level=logging.DEBUG)


# Endpoint for handling GET requests
@app.route("/predict", methods=["POST"])
def handle_get_request():
    logging.debug(f"Request received: {request}")
    logging.debug(f"Data received: {request.json}")

    data = pd.DataFrame(
        {header: request.json[header] for header in config["data_headers"]}
    )
    state = State(
        data,
        balance=request.json[BALANCE],
        contracts=request.json[CONTRACTS],
        entry_price=request.json[ENTRY_PRICE],
    )
    logging.debug(f"State constructed: {str(state)}")

    # Make a request to the model
    model_response = requests.post(
        "http://tensorflow-serve:8501/v1/models/prototype-V1:predict",
        json={"instances": [state.to_numpy().tolist()]},
    )
    logging.debug(f"Model Response: {model_response.json()}")

    predicted_result: float = model_response.json()

    return jsonify({"result": predicted_result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
