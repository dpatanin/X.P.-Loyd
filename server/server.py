import logging

import pandas as pd
import requests
import yaml
from flask import Flask, jsonify, request
from yaml.loader import FullLoader

with open("config.yaml") as f:
    config = yaml.load(f, Loader=FullLoader)

app = Flask(__name__)
# Configure logging
logging.basicConfig(filename="server.log", level=logging.DEBUG)


# Endpoint for handling GET requests
@app.route("/predict", methods=["POST"])
def handle_get_request():
    logging.debug(f"Request received: {request}")
    content = request.json
    logging.debug(f"Content received: {content}")

    data = pd.DataFrame({header: content[header] for header in config["data_headers"]})

    # Make a request to the model
    model_response = requests.post(
        "http://tensorflow-serve:8501/v1/models/model:predict",
        json={"instances": [data.to_numpy().tolist()]},
    )
    logging.debug(f"Model response: {model_response}")
    logging.debug(f"Model content: {model_response.json()}")

    prediction: float = model_response.json()["predictions"][0][0]
    logging.debug(f"Model prediction: {prediction}")

    response = {"prediction": prediction}

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
