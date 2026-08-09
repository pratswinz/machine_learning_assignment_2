"""
ML Assignment 2 - Model Training Script
Dataset: Breast Cancer Wisconsin (Diagnostic) - UCI ML Repository
Author: 2025AC05601
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET_COLUMN = "target"
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODEL_DIR)


def load_dataset():
    """Load Breast Cancer Wisconsin dataset from sklearn (UCI source)."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df[TARGET_COLUMN] = data.target
    return df, list(data.feature_names)


def evaluate_model(name, model, X_test, y_test):
    """Calculate all required evaluation metrics for a model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    return metrics, y_pred, y_prob


def main():
    print("Loading dataset...")
    df, feature_names = load_dataset()
    print(f"Total samples: {len(df)}, Features: {len(feature_names)}")

    # Split 75% train / 25% test (stratified)
    train_df, test_df = train_test_split(
        df, test_size=0.25, random_state=RANDOM_STATE, stratify=df[TARGET_COLUMN]
    )

    X_train = train_df[feature_names]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[feature_names]
    y_test = test_df[TARGET_COLUMN]

    # Save test data for Streamlit app upload
    test_csv_path = os.path.join(PROJECT_DIR, "test_data.csv")
    test_df.to_csv(test_csv_path, index=False)
    print(f"Saved test data to {test_csv_path}")

    # Define all 5 models (using pipelines where scaling is needed)
    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "kNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "Naive Bayes": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", GaussianNB()),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
    }

    all_metrics = []
    feature_names_path = os.path.join(MODEL_DIR, "feature_names.json")

    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f)

    print("\nTraining models...")
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)

        # Save model
        model_filename = name.lower().replace(" ", "_") + ".pkl"
        model_path = os.path.join(MODEL_DIR, model_filename)
        joblib.dump(model, model_path)

        metrics, y_pred, _ = evaluate_model(name, model, X_test, y_test)
        all_metrics.append(metrics)

        print(
            f"    Accuracy: {metrics['Accuracy']}, "
            f"AUC: {metrics['AUC']}, F1: {metrics['F1']}"
        )

    # Save metrics for README reference
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nAll metrics saved to {metrics_path}")
    print("Training complete!")


if __name__ == "__main__":
    main()
