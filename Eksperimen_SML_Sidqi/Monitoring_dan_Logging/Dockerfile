# Gunakan image yang sudah ada sebagai base
FROM qiamn/workflow-ci:latest

# Install dependensi tambahan
RUN pip install --no-cache-dir flask prometheus_client psutil

# Salin inference.py ke dalam container
COPY inference.py /app/inference.py