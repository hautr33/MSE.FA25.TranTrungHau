import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow import MlflowClient
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def tune_model(data_path):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Tuning_Logistic_Model")

    df = pd.read_csv(data_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {"C": [0.01, 0.1, 1.0, 10.0], "max_iter": [100, 200]}
    best_acc = 0
    best_run_id = None
    best_params = None

    for C in param_grid["C"]:
        for max_iter in param_grid["max_iter"]:
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            signature = infer_signature(X_test, preds)
            input_example = X_test.iloc[:1]

            with mlflow.start_run(run_name=f"Tuning_C{C}_maxiter{max_iter}") as run:
                mlflow.log_params({"C": C, "max_iter": max_iter})
                mlflow.log_metric("accuracy", acc)
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example
                )

                # Lưu lại best model run_id
                if acc > best_acc:
                    best_acc = acc
                    best_run_id = run.info.run_id
                    best_params = {"C": C, "max_iter": max_iter}

    print(f"✅ Best model accuracy: {best_acc} with params: {best_params}")
    return best_run_id

def register_best_model(run_id, model_name="best_classification_model"):
    client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
    
    # Đăng ký model vào Model Registry
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Gán alias 'production' cho version
    client.set_registered_model_alias(
        name=model_name,
        version=result.version,
        alias="production"
    )

    print(f"✅ Model registered as '{model_name}' version {result.version} and alias 'production' assigned.")

if __name__ == "__main__":
    data_path = "../data/classification_data.csv"

    best_run_id = tune_model(data_path)

    register_best_model(best_run_id)
