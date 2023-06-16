import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Endpoint for handling GET requests
@app.route('/', methods=['GET'])
def handle_get_request():
    # Read the content of the GET request
    content = request.args.get('content')

    # Make a request to the model
    model_response = requests.post('http://tensorflow-serve:8501/v1/models/prototype-V1:predict', json={'instances': [content]})

    # Print the model response
    print('Model Response:', model_response.json())

    # Extract the predicted result from the model response
    predicted_result = model_response.json()

    # Send the predicted result as a response back to the client
    return jsonify({'result': predicted_result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
