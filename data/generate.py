import os
import pandas as pd
from sklearn.datasets import make_classification

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
    output_dir = "../data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "classification_data.csv")

    df = generate_data()

    df.to_csv(output_path, index=False)

    print(f"✅ Data generated successfully!")
    print(f"🔹 Number of samples: {df.shape[0]}")
    print(f"🔹 Number of features (without target): {df.shape[1] - 1}")
    print(f"📁 File saved to: {output_path}")
