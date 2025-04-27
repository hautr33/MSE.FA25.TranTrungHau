# Makefile

# Config
MLFLOW_SERVER_HOST=127.0.0.1
MLFLOW_SERVER_PORT=5000
MLFLOW_BACKEND_URI=sqlite:///mlflow.db
MLFLOW_ARTIFACT_ROOT=./mlruns

# Các lệnh
start-mlflow:
	@echo "🚀 Starting MLflow Server..."
	mlflow server \
		--backend-store-uri $(MLFLOW_BACKEND_URI) \
		--default-artifact-root $(MLFLOW_ARTIFACT_ROOT) \
		--host 0.0.0.0 \
		--port $(MLFLOW_SERVER_PORT)

generate-data:
	@echo "🛠️  Generating data..."
	python data/generate.py

train:
	@echo "🎯 Training model..."
	python model/train.py

tune:
	@echo "🔍 Tuning hyperparameters and registering best model..."
	python model/tune.py

run-app:
	@echo "🌐 Running Flask app..."
	python app/app.py

# Lệnh tổng hợp
all: generate-data train tune run-app
