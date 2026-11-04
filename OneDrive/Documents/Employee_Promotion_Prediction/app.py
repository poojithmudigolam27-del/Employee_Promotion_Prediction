from pathlib import Path
import json
import warnings
import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "promotion_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_columns.pkl"
DATA_PATH = BASE_DIR / "data" / "employee_promotion_prediction.csv"
METRICS_PATH = BASE_DIR / "models" / "model_metrics.json"

st.set_page_config(
    page_title="Employee Promotion Prediction",
    page_icon="📈",
    layout="wide",
)


@st.cache_resource
def load_model_files():
    """Load the trained machine learning pipeline and required feature columns."""
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, list(feature_columns)


@st.cache_data
def load_form_metadata():
    """Discover dataset feature columns, categories, and numerical defaults."""
    if not DATA_PATH.exists():
        return [], [], [], {}, {}
    sample = pd.read_csv(DATA_PATH, nrows=5000)
    sample = sample.drop(columns=["promoted"], errors="ignore")
    categorical_columns = sample.select_dtypes(include=["object", "category", "string", "str"]).columns.tolist()
    numeric_columns = sample.select_dtypes(include="number").columns.tolist()

    category_options = {}
    for column in categorical_columns:
        values = sample[column].dropna().astype(str).unique().tolist()
        category_options[column] = sorted(values)

    numeric_defaults = {}
    for column in numeric_columns:
        median = sample[column].median()
        numeric_defaults[column] = float(median) if pd.notna(median) else 0.0

    return sample.columns.tolist(), categorical_columns, numeric_columns, category_options, numeric_defaults


@st.cache_data
def load_model_metrics():
    """Load model evaluation metrics if available."""
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# Verify required project paths
required_paths = [MODEL_PATH, FEATURES_PATH, DATA_PATH]
missing_paths = [str(path.relative_to(BASE_DIR)) for path in required_paths if not path.exists()]
if missing_paths:
    st.error("⚠️ Required project files are missing: " + ", ".join(missing_paths))
    st.info("Please ensure `models/promotion_model.pkl`, `models/feature_columns.pkl`, and `data/employee_promotion_prediction.csv` are present. You can train the model by running `python train.py`.")
    st.stop()

try:
    model, feature_columns = load_model_files()
    raw_columns, categorical_columns, numeric_columns, category_options, numeric_defaults = load_form_metadata()
    model_metrics = load_model_metrics()
except Exception as error:
    st.error(f"Could not load the model or dataset metadata: {error}")
    st.exception(error)
    st.stop()

# Initialize session state for form resets and persistent prediction output
if "form_version" not in st.session_state:
    st.session_state["form_version"] = 0
if "individual_result" not in st.session_state:
    st.session_state["individual_result"] = None

# Sidebar Overview
with st.sidebar:
    st.header("ℹ️ Model Overview")
    st.markdown(
        """
        **Algorithm:** Random Forest Classifier  
        **Pipeline:** Scikit-Learn `Pipeline`  
        **Preprocessing:**
        - Numerical Imputation (`median`)
        - Categorical One-Hot Encoding (`drop='first'`)
        - Balanced Class Weighting
        """
    )
    if model_metrics and "metrics" in model_metrics:
        m = model_metrics["metrics"]
        st.subheader("📊 Test Set Performance")
        st.metric("ROC AUC", f"{m.get('roc_auc', 0):.4f}")
        st.metric("Test Accuracy", f"{m.get('accuracy', 0) * 100:.1f}%")
        st.metric("Macro F1-Score", f"{m.get('macro_f1', 0):.4f}")
        st.metric("Promoted Class Recall", f"{m.get('promoted_recall', 0) * 100:.1f}%")
    st.divider()
    st.caption("Developed for HR Talent Management and Promotion Evaluation.")

# App Header
st.title("📈 Employee Promotion Prediction")
st.write(
    "Enter employee information to estimate whether an employee is likely "
    "to be promoted, using the trained machine-learning model pipeline."
)

tab_individual, tab_batch = st.tabs(["👤 Individual Employee Prediction", "📁 CSV Batch Prediction"])

# ==============================================================================
# TAB 1: INDIVIDUAL EMPLOYEE PREDICTION
# ==============================================================================
with tab_individual:
    col_hdr, col_reset = st.columns([4, 1])
    with col_hdr:
        st.subheader("Employee details")
    with col_reset:
        if st.button("🔄 Reset Fields", help="Reset all form fields to dataset median defaults"):
            st.session_state["form_version"] += 1
            st.session_state["individual_result"] = None
            st.rerun()

    current_version = st.session_state["form_version"]

    with st.form(f"employee_prediction_form_v{current_version}"):
        values = {}
        if categorical_columns:
            st.markdown("##### Categorical Details")
            category_layout = st.columns(2)
            for index, column in enumerate(categorical_columns):
                options = category_options.get(column, [])
                if options:
                    with category_layout[index % 2]:
                        values[column] = st.selectbox(
                            column.replace("_", " ").title(),
                            options=options,
                            key=f"cat_{column}_v{current_version}",
                        )

        if numeric_columns:
            st.markdown("##### Numeric Details")
            numeric_layout = st.columns(3)
            for index, column in enumerate(numeric_columns):
                default = numeric_defaults.get(column, 0.0)
                with numeric_layout[index % 3]:
                    if float(default).is_integer():
                        values[column] = st.number_input(
                            column.replace("_", " ").title(),
                            value=int(default),
                            step=1,
                            key=f"num_{column}_v{current_version}",
                        )
                    else:
                        values[column] = st.number_input(
                            column.replace("_", " ").title(),
                            value=float(default),
                            step=0.01,
                            format="%.2f",
                            key=f"num_{column}_v{current_version}",
                        )

        submitted = st.form_submit_button("Predict promotion", type="primary", use_container_width=True)

    if submitted:
        try:
            with st.spinner("Analyzing employee profile..."):
                input_df = pd.DataFrame([values])
                
                # Generate prediction and class probability
                prediction_class = int(model.predict(input_df)[0])
                probability = None
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_df)[0]
                    # Class 1 probability
                    probability = float(probs[1]) if len(probs) > 1 else float(probs[0])
                
                st.session_state["individual_result"] = {
                    "prediction_class": prediction_class,
                    "probability": probability,
                    "values": values,
                }
        except Exception as pred_error:
            st.error(f"An error occurred while generating prediction: {pred_error}")
            st.exception(pred_error)

    # Display prediction results if available in session state
    if st.session_state["individual_result"] is not None:
        res = st.session_state["individual_result"]
        pred = res["prediction_class"]
        prob = res["probability"]
        val = res["values"]

        st.divider()
        st.subheader("🎯 Prediction Result")

        result_col1, result_col2 = st.columns([3, 2])
        with result_col1:
            if pred == 1:
                st.success("### 🎉 Prediction: Likely to be Promoted")
                st.markdown(
                    "This employee profile displays strong performance and engagement indicators "
                    "consistent with historical promotion candidates."
                )
            else:
                st.warning("### 📋 Prediction: Not Likely to be Promoted")
                st.markdown(
                    "This employee profile currently falls below the historical threshold for promotion. "
                    "Focusing on key growth areas may improve future eligibility."
                )

            if prob is not None:
                prob_pct = prob * 100
                st.markdown(f"**Promotion Probability:** `{prob_pct:.1f}%`")
                st.progress(prob)

        with result_col2:
            st.markdown("##### Key Profile Indicators")
            st.metric(
                label="Performance Score",
                value=f"{val.get('performance_score', 'N/A')}",
            )
            st.metric(
                label="KPI Achievement",
                value=f"{val.get('kpi_achievement_percent', 'N/A'):.1f}%" if isinstance(val.get('kpi_achievement_percent'), (int, float)) else "N/A",
            )
            st.metric(
                label="Training Hours",
                value=f"{val.get('training_hours_last_year', 'N/A')}",
            )

        st.info(
            "ℹ️ **HR Advisory Notice:** This prediction is an algorithmic estimate based on historical data patterns "
            "and machine-learning evaluation. It is intended to support talent planning and should not replace official "
            "HR reviews or executive promotional committee decisions."
        )


# ==============================================================================
# TAB 2: CSV BATCH PREDICTION
# ==============================================================================
with tab_batch:
    st.subheader("Batch Promotion Assessment")
    st.write(
        "Upload a CSV file containing multiple employee records to generate "
        "promotional predictions and probabilities in bulk."
    )

    # Provide a sample template download button
    sample_file_path = BASE_DIR / "data" / "sample_batch_test.csv"
    if sample_file_path.exists():
        with open(sample_file_path, "rb") as sf:
            st.download_button(
                label="📥 Download Sample Batch CSV Template",
                data=sf.read(),
                file_name="sample_employee_batch.csv",
                mime="text/csv",
                help="Download a pre-formatted CSV template to test batch prediction immediately.",
            )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"**Uploaded records:** {len(batch_df)} rows, {len(batch_df.columns)} columns.")

            # Validate required columns
            # The model requires the 41 feature columns (ignoring employee_id or promoted if present)
            missing_features = [col for col in feature_columns if col not in batch_df.columns]

            if missing_features:
                st.error(
                    f"❌ Uploaded CSV is missing {len(missing_features)} required feature columns: "
                    + ", ".join(missing_features[:10])
                    + ("..." if len(missing_features) > 10 else "")
                )
                st.info("Ensure the uploaded CSV matches the employee dataset format. You can download the sample template above.")
            else:
                st.success("✅ All required feature columns are present!")
                st.dataframe(batch_df.head(5), use_container_width=True)

                if st.button("🚀 Run Batch Prediction", type="primary"):
                    with st.spinner(f"Evaluating {len(batch_df)} employee records..."):
                        # Predict using the identical saved pipeline
                        preds = model.predict(batch_df)
                        
                        output_df = batch_df.copy()
                        output_df["predicted_promoted"] = [
                            "Likely to be Promoted" if p == 1 else "Not Likely to be Promoted" for p in preds
                        ]

                        if hasattr(model, "predict_proba"):
                            probs = model.predict_proba(batch_df)[:, 1]
                            output_df["promotion_probability_%"] = (probs * 100).round(2)

                        st.subheader("📊 Batch Assessment Summary")
                        total_count = len(output_df)
                        promoted_count = int((preds == 1).sum())
                        not_promoted_count = total_count - promoted_count
                        promotion_rate = (promoted_count / total_count * 100) if total_count > 0 else 0

                        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
                        sum_col1.metric("Total Employees", total_count)
                        sum_col2.metric("Likely Promoted", f"{promoted_count} ({promotion_rate:.1f}%)")
                        sum_col3.metric("Not Likely Promoted", f"{not_promoted_count} ({100 - promotion_rate:.1f}%)")
                        if "promotion_probability_%" in output_df.columns:
                            sum_col4.metric("Average Probability", f"{output_df['promotion_probability_%'].mean():.1f}%")

                        st.subheader("📋 Prediction Results")
                        st.dataframe(output_df, use_container_width=True)

                        # Download button for batch results
                        csv_data = output_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Annotated Predictions CSV",
                            data=csv_data,
                            file_name="employee_promotion_predictions_results.csv",
                            mime="text/csv",
                        )

        except Exception as batch_error:
            st.error(f"Error processing CSV file: {batch_error}")
            st.exception(batch_error)
