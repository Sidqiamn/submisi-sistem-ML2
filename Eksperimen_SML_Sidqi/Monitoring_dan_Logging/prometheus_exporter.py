
from prometheus_client import Counter, Histogram, Gauge

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