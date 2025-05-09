# preprocessing/train_model_sidqi.py

import pandas as pd
import os
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
import joblib

# 1. Inisialisasi Dagshub untuk MLflow Tracking
dagshub.init(repo_owner='Sidqiamn', repo_name='Eksperimen_SML_Sidqi', mlflow=True)

# 2. Konfigurasi MLflow Tracking dengan Dagshub
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Sidqiamn/Eksperimen_SML_Sidqi.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "Sidqiamn"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://dagshub.com/api/v1/repo-buckets/s3/Sidqiamn"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MLFLOW_TRACKING_PUBLIC_KEY")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

# 3. Buat direktori models jika belum ada
if not os.path.exists('../models'):
    os.makedirs('../models')

# 4. Aktifkan autolog
mlflow.sklearn.autolog()

# 5. Load dataset yang sudah diproses
data = pd.read_csv('../dataset/processed/iris_processed.csv')

# 6. Pisahkan fitur dan target
X = data.drop('Species', axis=1)
y = data['Species']

# 7. Split data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Hyperparameter tuning dengan GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# 9. Mulai MLflow run
with mlflow.start_run() as run:
    # Latih model dengan GridSearchCV
    grid_search.fit(X_train, y_train)

    # Log parameter terbaik dari GridSearchCV
    best_params = grid_search.best_params_
    for param_name, param_value in best_params.items():
        mlflow.log_param(param_name, param_value)

    # Prediksi dan evaluasi model
    y_pred = grid_search.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)

    # Manual logging untuk metrik
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Log laporan klasifikasi sebagai artefak
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    # Simpan model terbaik
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, '../models/rf_model_sidqi.joblib')
    mlflow.log_artifact('../models/rf_model_sidqi.joblib')

    print("Akurasi model:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("\nLaporan Klasifikasi:\n", report)
    print("Model disimpan di ../models/rf_model_sidqi.joblib")