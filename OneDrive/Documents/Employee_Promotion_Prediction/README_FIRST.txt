# Employee Promotion Prediction System

This project predicts whether an employee is likely to be promoted using a machine-learning pipeline built with a balanced Random Forest Classifier and Streamlit.

## Project Structure
- `app.py` — Streamlit interactive web application for Individual and Batch CSV predictions.
- `train.py` — Production training pipeline that trains, evaluates, and exports the scikit-learn Pipeline.
- `test_app.py` — Automated test suite verifying predictions, batch processing, and missing-value resilience.
- `data/employee_promotion_prediction.csv` — Primary dataset (100,000 rows, 43 columns).
- `data/sample_batch_test.csv` — Ready-to-use sample batch CSV (10 records) for testing batch predictions.
- `models/promotion_model.pkl` — Trained end-to-end scikit-learn Pipeline (preprocessing + classifier).
- `models/feature_columns.pkl` — Serialized list of the 41 feature columns required by the model.
- `models/model_metrics.json` — Comprehensive model evaluation metrics on held-out test data.
- `notebooks/project.ipynb` — Exploratory data analysis and model prototyping notebook.
- `requirements.txt` — Project dependencies (Streamlit, scikit-learn, pandas, numpy, etc.).

## How to Run the Application

1. Open your terminal in this directory and activate the virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. Run the Streamlit web application:
   ```powershell
   python -m streamlit run app.py
   ```
3. Open your browser at the URL shown in the terminal (typically `http://localhost:8501`).

## How to Retrain the Model

To retrain the model and regenerate all artifacts in `models/`:
```powershell
python train.py
```

## How to Run Tests

To verify that all components, individual predictions, and batch predictions work correctly:
```powershell
python test_app.py
```

## Features
- **Individual Assessment:** Input employee metrics (performance, training, KPI achievement, leadership, etc.) and receive immediate promotion status, probability percentage, and tailored HR recommendations.
- **Batch CSV Processing:** Upload employee datasets, validate required feature columns, compute predictions and probabilities for all records, and download the annotated CSV.
- **Resilient Preprocessing:** Missing values are gracefully imputed, and unseen categorical levels are safely handled without application crashes.
