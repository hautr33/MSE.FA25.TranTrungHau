import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

def tune_model(data_path):
    df = pd.read_csv(data_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {"C": [0.01, 0.1, 1.0, 10.0], "max_iter": [100, 200]}

    best_acc = 0
    best_model = None
    best_params = None

    for C in param_grid["C"]:
        for max_iter in param_grid["max_iter"]:
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            with mlflow.start_run(run_name="Tuning_Run"):
                mlflow.log_params({"C": C, "max_iter": max_iter})
                mlflow.log_metric("accuracy", acc)
                mlflow.sklearn.log_model(model, artifact_path="model")

                if acc > best_acc:
                    best_acc = acc
                    best_model = model
                    best_params = {"C": C, "max_iter": max_iter}

    print(f"Best model accuracy: {best_acc} with params: {best_params}")
    return best_model

if __name__ == "__main__":
    tune_model("../data/classification_data.csv")