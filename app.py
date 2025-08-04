from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(_name_)

# Load model
model = joblib.load('train_model.pkl')  # Make sure this is the correct path

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

if _name_ == '_main_':
    app.run(debug=True)
