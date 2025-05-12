from flask import Flask, request, jsonify
import joblib
import pandas as pd
import time
import psutil
import os
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from prometheus_client.exposition import start_http_server

app = Flask(__name__)

model_path = "/app/models/rf_model_sidqi.joblib"
model = joblib.load(model_path)

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

active_sessions.set(0)

@app.route('/metrics')
def metrics():

    cpu_usage.set(psutil.cpu_percent(interval=1))
    memory_usage.set(psutil.virtual_memory().used)
    return generate_latest(REGISTRY), 200, {'Content-Type': 'text/plain; version=0.0.4'}

@app.route('/predict', methods=['POST'])
def predict():
    request_count.inc()  
    active_sessions.inc()  

    start_time = time.time()
    
    try:
        req_size = len(request.get_data())
        request_size.observe(req_size)

        data = request.get_json()
        if not data:
            raise ValueError("No input data provided")

        features = pd.DataFrame([data])
        if features.empty or features.shape[1] != 4:  
            raise ValueError("Invalid input data format")

        time.sleep(2)  

        inference_start = time.time()
        prediction = model.predict(features)[0]
        inference_duration = time.time() - inference_start
        inference_time.observe(inference_duration)

        response = {'prediction': int(prediction)} 
        resp_size = len(str(response).encode('utf-8'))
        response_size.observe(resp_size)

        prediction_success.inc()
        latency = time.time() - start_time
        request_latency.observe(latency)

        active_sessions.dec()  
        return jsonify(response)

    except Exception as e:
        prediction_error.inc()
        latency = time.time() - start_time
        request_latency.observe(latency)
        active_sessions.dec() 
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    start_http_server(8000)
    app.run(host='0.0.0.0', port=5000)