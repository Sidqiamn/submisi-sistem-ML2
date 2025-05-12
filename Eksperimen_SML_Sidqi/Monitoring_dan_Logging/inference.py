from flask import Flask, request, jsonify
import joblib
import pandas as pd
import time
import psutil
import os
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from prometheus_client.exposition import start_http_server

app = Flask(__name__)

# Load model
model_path = "/app/models/rf_model_sidqi.joblib"
model = joblib.load(model_path)

# Define Prometheus metrics
request_count = Counter('request_count', 'Total number of requests')
request_latency = Histogram('request_latency_seconds', 'Request latency in seconds')
prediction_success = Counter('prediction_success_count', 'Total number of successful predictions')
prediction_error = Counter('prediction_error_count', 'Total number of failed predictions')
inference_time = Histogram('model_inference_time_seconds', 'Model inference time in seconds')
cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
memory_usage = Gauge('memory_usage_bytes', 'Memory usage in bytes')
request_size = Histogram('request_size_bytes', 'Request size in bytes')
response_size = Histogram('response_size_bytes', 'Response size in bytes')
active_sessions = Gauge('active_sessions', 'Number of active sessions')

# Initialize active sessions
active_sessions.set(0)

@app.route('/metrics')
def metrics():
    # Update system metrics
    cpu_usage.set(psutil.cpu_percent(interval=1))
    memory_usage.set(psutil.virtual_memory().used)
    return generate_latest(REGISTRY), 200, {'Content-Type': 'text/plain; version=0.0.4'}

@app.route('/predict', methods=['POST'])
def predict():
    request_count.inc()  # Increment request count
    active_sessions.inc()  # Increment active sessions

    start_time = time.time()
    
    try:
        # Measure request size
        req_size = len(request.get_data())
        request_size.observe(req_size)

        # Get input data
        data = request.get_json()
        if not data:
            raise ValueError("No input data provided")

        # Prepare input for prediction
        features = pd.DataFrame([data])
        if features.empty or features.shape[1] != 4:  # Iris dataset has 4 features
            raise ValueError("Invalid input data format")

        # Simulate high latency
        time.sleep(2)  # Delay 2 detik untuk simulasi high latency

        # Measure inference time
        inference_start = time.time()
        prediction = model.predict(features)[0]
        inference_duration = time.time() - inference_start
        inference_time.observe(inference_duration)

        # Measure response size
        response = {'prediction': int(prediction)}  # Konversi ke int agar JSON valid
        resp_size = len(str(response).encode('utf-8'))
        response_size.observe(resp_size)

        # Log success
        prediction_success.inc()
        latency = time.time() - start_time
        request_latency.observe(latency)

        active_sessions.dec()  # Decrement active sessions
        return jsonify(response)

    except Exception as e:
        prediction_error.inc()
        latency = time.time() - start_time
        request_latency.observe(latency)
        active_sessions.dec()  # Decrement active sessions
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Start Prometheus metrics server on port 8000
    start_http_server(8000)
    # Run Flask app on port 5000
    app.run(host='0.0.0.0', port=5000)