"""
Gunicorn Production Configuration for Render Cloud Deployment
Optimized for running TensorFlow CNN models on cloud instances.
"""
import os

# Dynamic host and port binding (Render assigns PORT dynamically)
port = os.environ.get("PORT", "5000")
bind = f"0.0.0.0:{port}"

# Single worker with multithreading
# Essential for TensorFlow on cloud instances:
# 1. Loading multiple TensorFlow workers multiplies memory usage and causes OOM crashes.
# 2. A single worker with 4 threads keeps memory footprint stable (~250-350MB RSS)
#    well within cloud tier limits while servicing concurrent requests.
workers = 1
threads = 4

# Extended timeout (120 seconds) to accommodate cold-start TensorFlow model loading
timeout = 120

# Stream logs directly to stdout/stderr for Render log stream
accesslog = "-"
errorlog = "-"
loglevel = "info"
