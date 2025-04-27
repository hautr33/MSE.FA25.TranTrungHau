from flask import Flask, request, jsonify, render_template
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

app = Flask(__name__)

# Setup tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load model alias 'production'
model_uri = "models:/best_classification_model@production"
model = mlflow.sklearn.load_model(model_uri)

@app.route("/", methods=["GET", "POST"])
def home():
    # Generate random feature values between -2 and 2
    values = np.round(np.random.uniform(-2, 2, size=20), 2)
    return render_template("index.html", values=values, prediction=None)

@app.route("/predict_api", methods=["POST"])
def predict_api():
    input_json = request.get_json()
    input_data = pd.DataFrame([input_json])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][int(prediction)]
    return jsonify({
        "prediction": int(prediction),
        "probability": float(probability)
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005)
