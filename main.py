import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def run_sampling_analysis():
    df = pd.read_csv("Creditcard_data.csv")

    fraud = df[df["Class"] == 1]
    normal = df[df["Class"] == 0]

    fraud_oversampled = fraud.sample(len(normal), replace=True, random_state=42)
    balanced_df = pd.concat([fraud_oversampled, normal]).reset_index(drop=True)

    print(f"Dataset Balanced: {len(balanced_df)} total rows ({len(normal)} per class).")

    # Formula: n = (Z^2 * p * (1-p)) / E^2
    z = 1.96
    p = 0.5
    e = 0.05
    n = math.ceil((z**2 * p * (1 - p)) / (e**2))
    print(f"Calculated Sample Size (n): {n}")

    samples = {}

    # Simple Random Sampling
    samples["Sampling1"] = balanced_df.sample(n=n, random_state=42).copy()

    # Systematic Sampling
    step = len(balanced_df) // n
    samples["Sampling2"] = balanced_df.iloc[::step].head(n).copy()

    # Stratified Sampling (Maintaining 50/50 ratio)
    sample3_class0 = balanced_df[balanced_df["Class"] == 0].sample(
        n // 2, random_state=42
    )
    sample3_class1 = balanced_df[balanced_df["Class"] == 1].sample(
        n // 2, random_state=42
    )
    samples["Sampling3"] = pd.concat([sample3_class0, sample3_class1]).reset_index(
        drop=True
    )

    # Cluster Sampling
    temp_df = balanced_df.copy()
    cluster_size = 10
    temp_df["ClusterID"] = np.arange(len(temp_df)) // cluster_size
    total_clusters = temp_df["ClusterID"].nunique()
    selected_clusters = np.random.choice(
        range(total_clusters), size=max(1, n // cluster_size), replace=False
    )

    cluster_sample = temp_df[temp_df["ClusterID"].isin(selected_clusters)].copy()
    samples["Sampling4"] = cluster_sample.drop(columns=["ClusterID"])

    # Bootstrap Sampling (Random sampling with replacement)
    samples["Sampling5"] = balanced_df.sample(n=n, replace=True, random_state=42).copy()

    models = {
        "M1": LogisticRegression(max_iter=2000),
        "M2": DecisionTreeClassifier(random_state=42),
        "M3": RandomForestClassifier(random_state=42),
        "M4": SVC(kernel="linear"),
        "M5": KNeighborsClassifier(),
    }

    results = pd.DataFrame(index=models.keys(), columns=samples.keys())

    for s_name, s_data in samples.items():
        if "Class" not in s_data.columns:
            print(f"Warning: 'Class' column missing in {s_name}. Skipping...")
            continue

        X = s_data.drop(columns=["Class"], errors="ignore")
        y = s_data["Class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        for m_name, model in models.items():
            try:
                # Train and predict
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                results.loc[m_name, s_name] = round(acc, 4)
            except Exception as e:
                print(f"Error training {m_name} on {s_name}: {e}")
                results.loc[m_name, s_name] = np.nan

    print(results)
    results.to_csv("sampling_model_comparison.csv")


if __name__ == "__main__":
    run_sampling_analysis()
