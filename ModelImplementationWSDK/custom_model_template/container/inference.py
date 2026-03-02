import flask
#import pickle5 as pickle
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask app
app = Flask(__name__)

# Load the model
with open('/opt/ml/model/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/ping', methods=['GET'])
def ping():
    """
    Determine if the container is working and healthy.
    """
    return flask.Response(response='\n', status=200, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """
    Do an inference on a single batch of data.
    """
    if flask.request.content_type == 'application/json':
        data = flask.request.get_json()
        
        # Assuming the input is a list of features
        input_data = pd.DataFrame(data['inputs'])
        
        # Make prediction
        prediction = model.predict_proba(input_data)[:, 1].tolist()
        
        return jsonify({'prediction': prediction})
    else:
        return flask.Response(response='This predictor only supports JSON data', status=415, mimetype='text/plain')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)