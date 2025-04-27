# 🚀 MLOps Project: Logistic Regression Classification

## 📖 Introduction

This project demonstrates a basic MLOps workflow:

- Generate synthetic classification data (make_classification)

- Train a Logistic Regression model

- Tune hyperparameters (C, max_iter)

- Log all experiments to MLflow Tracking Server

- Automatically register the best model to Model Registry (alias: production)

- Deploy a Flask App serving a prediction API

## 📂 Project Structure
```
.
├── app
│   ├── app.py
│   └── templates
│       └── index.html
├── data
│   ├── classification_data.csv
│   └── generate.py
├── model
│   ├── train.py
│   └── tune.py
├── config.yaml
├── makefile
├── mlflow.db
├── mlruns
├── README.md
└── requirements.txt
```
## ⚙️ Setup and Run Instructions
### 1. Install dependencies
```
pip install -r requirements.txt
```
### 2. Start MLflow Tracking Server
```
make start-mlflow
```
✅ MLflow UI available at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 3. Generate classification data
```
make generate-data
```
✅ The classification_data.csv file will be saved inside the data/ folder.

### 4. Train a basic model
```
make train
```
✅ Train a simple Logistic Regression model and log it to MLflow.

### 5. Tune hyperparameters and register the best model
```
make tune
```
✅ Automatically select the best model and register it into the Model Registry with alias production.

### 6. Run Flask App to serve predictions
```
make run-app
```
✅ Flask App available at: [http://127.0.0.1:5005](http://127.0.0.1:5005)

## 🌟 Notes

- MLflow server must be running during training, tuning, and serving.

- If the server is down, rerun make start-mlflow before continuing.

- The Flask App loads the model from MLflow Model Registry alias production.

## 📈 Overall Workflow (Flow)

[1] Generate data ➔ [2] Train ➔ [3] Tune ➔ [4] Register model ➔ [5] Serve API ➔ [6] Send predict requests

## 🔥 Future Enhancements

- Add CI/CD pipelines to auto-train and auto-deploy models.

- Deploy MLflow Tracking and Model Registry using AWS S3 or Azure Blob Storage.

- Dockerize the entire system for easy deployment.