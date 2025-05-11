# Gunakan base image Python
FROM python:3.12.7-slim

# Set working directory
WORKDIR /app

# Salin file yang diperlukan
COPY modelling.py .
COPY dataset/ ./dataset/
COPY models/ ./models/

# Install dependensi secara langsung
RUN pip install --no-cache-dir pandas scikit-learn mlflow==2.14.3 joblib dagshub

# Perintah default
CMD ["python", "modelling.py"]