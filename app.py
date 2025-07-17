from flask import Flask, request, jsonify
import torch
import numpy as np
from predict import predict_future
from flask import Flask, jsonify, request



app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
    days = int(request.args.get("days", 10))  # Default: 10 days
    predictions = predict_future(days)
    return jsonify({"predictions": predictions.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

