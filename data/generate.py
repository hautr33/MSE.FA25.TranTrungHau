from sklearn.datasets import make_classification
import pandas as pd

def generate_data(n_samples=1000, n_features=20, n_classes=2, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        n_redundant=2,
        n_classes=n_classes,
        random_state=random_state
    )
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    data['target'] = y
    return data

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("../data/classification_data.csv", index=False)