from lib.constants import BALANCE, ENTRY_PRICE, CONTRACTS, ACTION_STAY, ACTION_EXIT
from lib.action_space import ActionSpace
from lib.state import State
from flask import Flask, request, jsonify
import requests
import logging
import pandas as pd

import yaml
from yaml.loader import FullLoader

with open("config.yaml") as f:
    config = yaml.load(f, Loader=FullLoader)

app = Flask(__name__)
# Configure logging
logging.basicConfig(filename="server.log", level=logging.DEBUG)

action_space = ActionSpace(
    threshold=config["action_space"]["threshold"],
    price_per_contract=config["tick_value"],
    limit=config["action_space"]["trade_limit"],
    intrinsic_fac=config["reward_factors"]["intrinsic"],
)


# Endpoint for handling GET requests
@app.route("/predict", methods=["POST"])
def handle_get_request():
    logging.debug(f"Request received: {request}")
    content = request.json
    logging.debug(f"Content received: {content}")

    data = pd.DataFrame({header: content[header] for header in config["data_headers"]})
    state = State(
        data,
        balance=content[BALANCE],
        contracts=content[CONTRACTS],
        entry_price=content[ENTRY_PRICE],
        tick_size=content["tick_size"],
        tick_value=content["tick_value"]
    )
    logging.debug(f"State constructed: {str(state)}")

    # Make a request to the model
    model_response = requests.post(
        "http://tensorflow-serve:8501/v1/models/model:predict",
        json={"instances": [state.to_numpy().tolist()]},
    )
    logging.debug(f"Model response: {model_response}")
    logging.debug(f"Model content: {model_response.json()}")

    prediction: float = model_response.json()["predictions"][0][0]
    logging.debug(f"Model prediction: {prediction}")

    # Take action, calculate amount inversely & return response
    amount = action_space.calc_trade_amount(prediction, state)
    action = action_space.take_action(prediction, state)[1]
    if action in [ACTION_EXIT, ACTION_STAY]:
        amount = 0

    response = {"action": action, "amount": amount}

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
