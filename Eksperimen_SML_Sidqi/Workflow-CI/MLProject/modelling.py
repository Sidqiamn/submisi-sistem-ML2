

import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
import joblib

print("MLFLOW_TRACKING_URI:", os.getenv("MLFLOW_TRACKING_URI", "Not set"))
print("MLFLOW_TRACKING_USERNAME:", os.getenv("MLFLOW_TRACKING_USERNAME", "Not set"))
print("MLFLOW_TRACKING_PASSWORD:", os.getenv("MLFLOW_TRACKING_PASSWORD", "Not set"))
print("MLFLOW_S3_ENDPOINT_URL:", os.getenv("MLFLOW_S3_ENDPOINT_URL", "Not set"))
print("AWS_ACCESS_KEY_ID:", os.getenv("AWS_ACCESS_KEY_ID", "Not set"))
print("AWS_SECRET_ACCESS_KEY:", os.getenv("AWS_SECRET_ACCESS_KEY", "Not set"))
print("AWS_DEFAULT_REGION:", os.getenv("AWS_DEFAULT_REGION", "Not set"))

if not os.path.exists('models'):
    os.makedirs('models')

try:
    mlflow.sklearn.autolog()
    print("Autolog activated successfully.")
except Exception as e:
    print(f"Error activating autolog: {e}")
    raise

try:
    dataset_path = "dataset/processed/iris_processed.csv"
    data = pd.read_csv(dataset_path)
    print("Kolom dalam dataset:", data.columns)
except FileNotFoundError:
    print(f"File 'iris_processed.csv' tidak ditemukan di path: {dataset_path}")
    raise

print("Cek nilai NaN di dataset:")
print(data.isna().sum())

if data['Species'].isna().sum() > 0:
    print(f"Terdapat {data['Species'].isna().sum()} nilai NaN di kolom 'Species'.")
    data = data.dropna(subset=['Species'])
    print("Baris dengan NaN di kolom 'Species' telah dihapus.")
else:
    print("Tidak ada nilai NaN di kolom 'Species'.")

if data.drop('Species', axis=1).isna().sum().sum() > 0:
    print("Terdapat nilai NaN di fitur. Mengisi dengan median...")
    data = data.fillna(data.median(numeric_only=True))
else:
    print("Tidak ada nilai NaN di fitur.")

X = data.drop('Species', axis=1)
y = data['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

try:
    with mlflow.start_run() as run:
        # Latih model dengan GridSearchCV
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)

        y_pred = grid_search.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        best_model = grid_search.best_estimator_
        joblib.dump(best_model, 'models/rf_model_sidqi.joblib')
        mlflow.log_artifact('models/rf_model_sidqi.joblib')

        print("Akurasi model:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("\nLaporan Klasifikasi:\n", report)
        print("Model disimpan di models/rf_model_sidqi.joblib")
except Exception as e:
    print(f"Error saat menjalankan MLflow run: {e}")
    raise