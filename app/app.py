from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd

app = Flask(__name__)

# Load best model from MLflow
model_uri = "models:/best_classification_model/Production"
model = mlflow.sklearn.load_model(model_uri)

@app.route("/predict", methods=["POST"])
def predict():
    input_json = request.get_json()
    input_data = pd.DataFrame([input_json])
    prediction = model.predict(input_data)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)