# server.py

"""
In this file, we will use the flask web framework 
to handle the POST requests that we will get from the request.py.
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import sys

# Initialize the Flask app
app = Flask(__name__)


# Use the route() decorator to tell Flask what URL should trigger our function.
@app.route("/")
def home_page():
    print(**locals())
    return render_template("index.html", **locals())


# "python -m flask --app server run" to run


# Load the ML model first
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/predict', methods=['POST'])
def predict():
    # data = request.get_json(force=True)
    # prediction = model.predict([[np.array(data['exp'])]])
    # output = prediction[0]
    YearsExp = request.form["Exp"]
    
    # we're changing this here because we need to transform the input data to something that the ML model accepts
    YearsExp = [[float(request.form.get('Exp'))]]
    result = float(model.predict(YearsExp))
# (YearsExp[0])[0])[0]
    return render_template("index.html", **locals())

@app.route('/api', methods=['POST'])
def predict_from_request():
    data = request.get_json(force=True)
    # we're changing this here because we need to transform the input data to something that the ML model accepts
    prediction = model.predict([[np.array(data['exp'])]])
    output = prediction[0]
    return jsonify(output)


# if we're running this file directly, it will run the Flask server
if __name__ == '__main__':
    app.run(port=5002, debug=True)
