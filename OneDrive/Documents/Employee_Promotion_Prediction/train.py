"""
train.py
====================================================================
Employee Promotion Prediction - Model Training and Export Pipeline

This script trains a balanced Random Forest classification pipeline
for predicting employee promotions. It handles:
- Stratified train/test splitting (80/20)
- Missing value imputation
- Categorical one-hot encoding (handle_unknown='ignore', drop='first')
- Class imbalance correction using balanced tree weights
- Model evaluation (Accuracy, ROC AUC, Precision, Recall, F1, Confusion Matrix)
- Serialization of the complete end-to-end Pipeline and feature column metadata
====================================================================
"""

import os
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "employee_promotion_prediction.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "promotion_model.pkl"
FEATURES_PATH = MODELS_DIR / "feature_columns.pkl"
METRICS_PATH = MODELS_DIR / "model_metrics.json"


def train_and_export_model():
    print(f"Loading dataset from: {DATA_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

    target_col = "promoted"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset!")

    # Exclude ID and target from predictive features to prevent data leakage
    drop_cols = [target_col]
    if "employee_id" in df.columns:
        drop_cols.append("employee_id")

    X = df.drop(columns=drop_cols)
    y = df[target_col]

    # Identify categorical and numerical columns
    categorical_features = X.select_dtypes(include=["object", "category", "string", "str"]).columns.tolist()
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    feature_columns = categorical_features + numeric_features

    print(f"Features: {len(categorical_features)} categorical, {len(numeric_features)} numerical (Total: {len(feature_columns)})")
    print(f"Target distribution:\n{y.value_counts(normalize=True).to_dict()}")

    # Perform stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train split size: {len(X_train)}, Test split size: {len(X_test)}")

    # Preprocessing pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",  # Safely ignores extra columns such as employee_id
    )

    # High-performance Random Forest with balanced class weights
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    print("Fitting model pipeline on training data...")
    full_pipeline.fit(X_train, y_train)

    print("Evaluating model on held-out test data...")
    y_pred = full_pipeline.predict(X_test)
    y_prob = full_pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    prec_1 = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec_1 = float(recall_score(y_test, y_pred, pos_label=1))
    f1_1 = float(f1_score(y_test, y_pred, pos_label=1))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS (HELD-OUT TEST SET)")
    print("=" * 50)
    print(f"Accuracy:          {acc * 100:.2f}%")
    print(f"Balanced Accuracy: {bal_acc * 100:.2f}%")
    print(f"ROC AUC:           {roc_auc:.4f}")
    print(f"Macro F1-Score:    {macro_f1:.4f}")
    print(f"Promoted Class (1) Precision: {prec_1 * 100:.2f}%")
    print(f"Promoted Class (1) Recall:    {rec_1 * 100:.2f}%")
    print(f"Promoted Class (1) F1-Score:  {f1_1:.4f}")
    print("\nConfusion Matrix:")
    print(f"  TN: {cm[0][0]:6d} | FP: {cm[0][1]:6d}")
    print(f"  FN: {cm[1][0]:6d} | TP: {cm[1][1]:6d}")
    print("\nClassification Report:\n" + classification_report(y_test, y_pred))
    print("=" * 50)

    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save Pipeline and feature list
    print(f"Saving full pipeline to {MODEL_PATH}...")
    joblib.dump(full_pipeline, MODEL_PATH)

    print(f"Saving feature columns to {FEATURES_PATH}...")
    joblib.dump(feature_columns, FEATURES_PATH)

    metrics_data = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 16,
        "class_weight": "balanced",
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_count": len(feature_columns),
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "metrics": {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "roc_auc": roc_auc,
            "macro_f1": macro_f1,
            "promoted_precision": prec_1,
            "promoted_recall": rec_1,
            "promoted_f1": f1_1,
            "confusion_matrix": cm,
        },
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved evaluation metrics to {METRICS_PATH}")

    # Self-test: reload saved artifacts and test inference on a sample
    print("Verifying saved artifacts...")
    loaded_pipe = joblib.load(MODEL_PATH)
    loaded_cols = joblib.load(FEATURES_PATH)

    sample_test = X_test.iloc[:5]
    preds = loaded_pipe.predict(sample_test)
    probs = loaded_pipe.predict_proba(sample_test)[:, 1]

    print("Sample test successful! Predictions:", preds.tolist())
    print("Sample test probabilities:", [round(p, 4) for p in probs])
    print("Training and verification complete!")


if __name__ == "__main__":
    train_and_export_model()
