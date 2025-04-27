import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature

def train_model():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Simple_Classification_Experiment")

    # Sinh dữ liệu phân loại
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    y = pd.Series(y, name="target")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    signature = infer_signature(X_test, preds)
    input_example = X_test.iloc[:1]

    with mlflow.start_run(run_name="Simple_Logistic_Model"):
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

    print(f"Training complete. Accuracy: {acc}")

if __name__ == "__main__":
    train_model()
