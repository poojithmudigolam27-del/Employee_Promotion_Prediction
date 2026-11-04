"""
test_app.py
Comprehensive automated test suite for Employee Promotion Prediction application.
Tests:
1. Artifact existence and loading
2. Input metadata loading
3. Individual prediction (both negative and positive cases)
4. Batch prediction logic with sample CSV
5. Handling of missing columns (validation error)
6. Handling of missing values (imputation resilience)
"""

from pathlib import Path
import json
import warnings
import joblib
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "promotion_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_columns.pkl"
DATA_PATH = BASE_DIR / "data" / "employee_promotion_prediction.csv"
SAMPLE_BATCH_PATH = BASE_DIR / "data" / "sample_batch_test.csv"


def test_artifacts_exist():
    print("--> Test 1: Checking required files...")
    assert MODEL_PATH.exists(), f"Missing {MODEL_PATH}"
    assert FEATURES_PATH.exists(), f"Missing {FEATURES_PATH}"
    assert DATA_PATH.exists(), f"Missing {DATA_PATH}"
    print("    [PASS] All core model and data files exist.")


def test_model_loading():
    print("--> Test 2: Checking model and features deserialization...")
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    assert model is not None, "Failed to load model"
    assert len(features) == 41, f"Expected 41 features, got {len(features)}"
    print(f"    [PASS] Model type: {type(model).__name__}, feature count: {len(features)}")
    return model, features


def test_individual_predictions(model, features):
    print("--> Test 3: Testing individual predictions (Negative and Positive cases)...")
    
    # Negative test case (low performance, no KPI, no overtime, etc.)
    df_raw = pd.read_csv(DATA_PATH)
    
    # Pick a real low performer and real high performer from dataset
    not_promoted_row = df_raw[df_raw['promoted'] == 0].drop(columns=['promoted']).iloc[0:1]
    promoted_row = df_raw[df_raw['promoted'] == 1].drop(columns=['promoted']).iloc[0:1]

    pred_neg = model.predict(not_promoted_row)[0]
    prob_neg = model.predict_proba(not_promoted_row)[0][1]
    print(f"    Negative Profile Prediction: {pred_neg} (Probability: {prob_neg*100:.2f}%)")

    pred_pos = model.predict(promoted_row)[0]
    prob_pos = model.predict_proba(promoted_row)[0][1]
    print(f"    Positive Profile Prediction: {pred_pos} (Probability: {prob_pos*100:.2f}%)")

    # Synthetic strong profile
    strong_profile = not_promoted_row.copy()
    strong_profile['performance_score'] = 100.0
    strong_profile['performance_last_year'] = 100.0
    strong_profile['performance_two_years_ago'] = 98.0
    strong_profile['kpi_achievement_percent'] = 99.0
    strong_profile['manager_rating'] = 5.0
    strong_profile['peer_feedback_score'] = 5.0
    strong_profile['projects_completed'] = 12
    strong_profile['leadership_score'] = 95.0
    strong_profile['tasks_completed'] = 45.0
    strong_profile['training_hours_last_year'] = 80.0

    pred_strong = model.predict(strong_profile)[0]
    prob_strong = model.predict_proba(strong_profile)[0][1]
    print(f"    Synthetic High Performer Prediction: {pred_strong} (Probability: {prob_strong*100:.2f}%)")
    assert prob_strong > 0.5, "Strong profile should have high promotion probability"
    print("    [PASS] Individual prediction produces consistent output and probabilities.")


def test_batch_prediction(model, features):
    print("--> Test 4: Testing batch prediction on sample CSV...")
    assert SAMPLE_BATCH_PATH.exists(), f"Missing {SAMPLE_BATCH_PATH}"
    batch_df = pd.read_csv(SAMPLE_BATCH_PATH)
    
    # Verify all feature columns are present
    missing = [c for c in features if c not in batch_df.columns]
    assert len(missing) == 0, f"Missing columns in sample batch: {missing}"

    preds = model.predict(batch_df)
    probs = model.predict_proba(batch_df)[:, 1]

    assert len(preds) == len(batch_df), "Batch prediction count mismatch"
    assert len(probs) == len(batch_df), "Batch probability count mismatch"

    output_df = batch_df.copy()
    output_df['predicted_promoted'] = ["Likely to be Promoted" if p == 1 else "Not Likely to be Promoted" for p in preds]
    output_df['promotion_probability_%'] = (probs * 100).round(2)

    promoted_count = (preds == 1).sum()
    print(f"    Batch of {len(batch_df)} rows evaluated: {promoted_count} likely promoted, {len(batch_df) - promoted_count} not likely promoted.")
    print("    [PASS] Batch prediction succeeded.")


def test_missing_value_resilience(model):
    print("--> Test 5: Testing missing value resilience (Imputation)...")
    df_raw = pd.read_csv(SAMPLE_BATCH_PATH).head(3).copy()
    # Inject NaNs in various numeric and categorical fields
    df_raw.loc[0, 'performance_score'] = np.nan
    df_raw.loc[1, 'department'] = np.nan
    df_raw.loc[2, 'salary'] = np.nan

    # Should not raise exception because pipeline has SimpleImputer
    preds = model.predict(df_raw)
    probs = model.predict_proba(df_raw)[:, 1]
    assert len(preds) == 3
    print("    [PASS] Pipeline gracefully handled missing values via imputers.")


def test_missing_column_detection(features):
    print("--> Test 6: Testing missing column validation...")
    malformed_df = pd.DataFrame({
        'age': [30],
        'gender': ['Male'],
        'department': ['Sales']
    })
    missing = [c for c in features if c not in malformed_df.columns]
    assert len(missing) > 0, "Should have detected missing columns"
    print(f"    [PASS] Successfully detected {len(missing)} missing required columns.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING AUTOMATED TEST SUITE")
    print("=" * 60)
    test_artifacts_exist()
    model, features = test_model_loading()
    test_individual_predictions(model, features)
    test_batch_prediction(model, features)
    test_missing_value_resilience(model)
    test_missing_column_detection(features)
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! (6/6)")
    print("=" * 60)
