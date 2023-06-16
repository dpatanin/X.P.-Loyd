import requests
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)
# Configure logging
logging.basicConfig(filename='server.log', level=logging.DEBUG)

# Endpoint for handling GET requests
@app.route('/predict', methods=['POST'])
def handle_get_request():
    content = request.json
    logging.debug(f'Received content: {content}')

    # Make a request to the model
    model_response = requests.post('http://tensorflow-serve:8501/v1/models/prototype-V1:predict', json={'instances': [content]})

    logging.debug(f'Model Response: {model_response.json()}')

    predicted_result = model_response.json()

    return jsonify({'result': predicted_result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
