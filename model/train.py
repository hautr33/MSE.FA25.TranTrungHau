import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

def train_model(data_path):
    df = pd.read_csv(data_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    with mlflow.start_run(run_name="Simple_Logistic_Model"):
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_test.iloc[:1])

    print(f"Training complete with accuracy: {acc}")

if __name__ == "__main__":
    train_model("../data/classification_data.csv")