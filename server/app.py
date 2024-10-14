from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the scaler and model
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("kmeans_model.pkl", "rb") as model_file:
    kmeans_model = pickle.load(model_file)


@app.route("/predict-segment", methods=["POST"])
def predict_segment():
    data = request.json
    # Extract features from the request
    features = np.array(
        [
            [
                data["Age"],
                data["Annual_Income"],
                data["Spending_Score"],
                data["Loyalty_Status"],
            ]
        ]
    )

    # Scale the features
    scaled_features = scaler.transform(features)

    # Predict the cluster
    cluster = kmeans_model.predict(scaled_features)

    # Return the predicted cluster
    return jsonify({"cluster": int(cluster[0])})


if __name__ == "__main__":
    app.run(debug=True)
