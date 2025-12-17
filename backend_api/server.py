"""
Backend API Server for Fraud Detection Frontend
Exposes Redis data and Prometheus metrics via REST API
"""
import json
import os
from flask import Flask, jsonify, Response, request
from flask_cors import CORS
import redis
import requests
from typing import List, Dict, Any

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
PROMETHEUS_URL = os.getenv('PROMETHEUS_URL', 'http://localhost:9090')

# Initialize Redis connection
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def prom_instant(metric: str) -> float:
    """Query Prometheus for instant metric value"""
    try:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': metric}, timeout=2)
        data = r.json()
        res = data.get('data', {}).get('result', [])
        if not res:
            return 0.0
        return float(res[0]['value'][1])
    except Exception as e:
        print(f"Error querying Prometheus: {e}")
        return 0.0


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        redis_client.ping()
        return jsonify({'status': 'healthy', 'redis': 'connected'})
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500


@app.route('/api/predictions/latest', methods=['GET'])
def get_latest_predictions():
    """Get latest predictions from Redis"""
    try:
        limit = int(request.args.get('limit', 100))
        items = redis_client.lrange('predictions:latest', 0, limit - 1)
        predictions = []
        for item in reversed(items):  # oldest first
            try:
                predictions.append(json.loads(item))
            except Exception:
                continue
        return jsonify({'predictions': predictions, 'count': len(predictions)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predictions/stream', methods=['GET'])
def stream_predictions():
    """Server-Sent Events stream for real-time predictions"""
    def generate():
        last_count = 0
        while True:
            try:
                current_count = redis_client.llen('predictions:latest')
                if current_count > last_count:
                    # Get new items
                    new_items = redis_client.lrange('predictions:latest', last_count, current_count - 1)
                    for item in new_items:
                        try:
                            prediction = json.loads(item)
                            yield f"data: {json.dumps(prediction)}\n\n"
                        except Exception:
                            continue
                    last_count = current_count
                else:
                    # Send heartbeat
                    yield ": heartbeat\n\n"
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                break
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get Prometheus metrics"""
    try:
        throughput = prom_instant('throughput_msgs_per_sec')
        latency = prom_instant('spark_processing_latency_ms')
        infer_rate = prom_instant('rate(model_inference_latency_ms_count[5s])')
        acc_tp = prom_instant('detection_true_positive_total')
        acc_fp = prom_instant('detection_false_positive_total')
        acc_fn = prom_instant('detection_false_negative_total')
        
        precision = (acc_tp / (acc_tp + acc_fp)) if (acc_tp + acc_fp) > 0 else 0.0
        recall = (acc_tp / (acc_tp + acc_fn)) if (acc_tp + acc_fn) > 0 else 0.0
        
        return jsonify({
            'throughput': round(throughput, 2),
            'latency': round(latency, 1),
            'inference_rate': round(infer_rate, 2),
            'precision': round(precision, 2),
            'recall': round(recall, 2),
            'true_positives': int(acc_tp),
            'false_positives': int(acc_fp),
            'false_negatives': int(acc_fn)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/transaction/<transaction_id>', methods=['GET'])
def get_transaction(transaction_id: str):
    """Get specific transaction by ID"""
    try:
        key = f"txn:{transaction_id}"
        data = redis_client.hgetall(key)
        if not data:
            return jsonify({'error': 'Transaction not found'}), 404
        
        # Parse JSON values
        parsed_data = {}
        for k, v in data.items():
            try:
                parsed_data[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                parsed_data[k] = v
        
        return jsonify(parsed_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print(f"Starting backend API server...")
    print(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
    print(f"Prometheus: {PROMETHEUS_URL}")
    app.run(host='0.0.0.0', port=5000, debug=True)

