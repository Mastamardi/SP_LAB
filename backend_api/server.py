"""
Backend API Server for Fraud Detection Frontend
Exposes Redis data and Prometheus metrics via REST API.
"""

import json
import os
import time
from typing import Any, Dict

from flask import Flask, jsonify, Response, request
from flask_cors import CORS
import redis
import requests


app = Flask(__name__)
CORS(app)  # Enable CORS for frontend (adjust in production)

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

# Initialize Redis client (connection is lazy until first command)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def prom_instant(metric: str) -> float:
    """Query Prometheus for an instant metric value."""
    try:
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": metric},
            timeout=2,
        )
        data = r.json()
        res = data.get("data", {}).get("result", [])
        if not res:
            return 0.0
        return float(res[0]["value"][1])
    except Exception:
        # For robustness, just return 0.0 on errors
        return 0.0


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    try:
        redis_client.ping()
        return jsonify({"status": "healthy", "redis": "connected"})
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route("/api/predictions/latest", methods=["GET"])
def get_latest_predictions():
    """Get latest predictions from Redis."""
    try:
        limit = int(request.args.get("limit", 100))
        items = redis_client.lrange("predictions:latest", 0, limit - 1)
        predictions = []
        for item in reversed(items):  # oldest first
            try:
                predictions.append(json.loads(item))
            except Exception:
                continue
        return jsonify({"predictions": predictions, "count": len(predictions)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predictions/stream", methods=["GET"])
def stream_predictions():
    """Server-Sent Events stream for real-time predictions."""

    def generate():
        # Start from the current end of the list so each new connection
        # only sees NEW predictions (not the entire history)
        last_count = redis_client.llen("predictions:latest")
        error_count = 0
        max_errors = 10

        while True:
            try:
                # Check Redis connection
                redis_client.ping()
                error_count = 0  # reset error count if OK

                current_count = redis_client.llen("predictions:latest")
                if current_count > last_count:
                    # Get new items - since LPUSH adds to left (index 0), newest items are at the beginning
                    # Read from index 0 to (num_new_items - 1) to get the newest items first
                    num_new = current_count - last_count
                    new_items = redis_client.lrange(
                        "predictions:latest", 0, num_new - 1
                    )
                    # Send newest first (index 0 is newest with LPUSH)
                    # This ensures the most recent transaction appears at the top of the UI
                    for item in new_items:
                        try:
                            prediction = json.loads(item)
                            yield f"data: {json.dumps(prediction)}\n\n"
                        except Exception:
                            continue
                    last_count = current_count
                else:
                    # Heartbeat to keep connection alive - check more frequently
                    yield ": heartbeat\n\n"
                    time.sleep(0.5)  # Check every 0.5 seconds for new transactions

            except redis.ConnectionError:
                error_count += 1
                if error_count <= max_errors:
                    yield (
                        "event: error\n"
                        f"data: {json.dumps({'error': 'Redis not connected', 'message': 'Waiting for Redis connection...'})}\n\n"
                    )
                    time.sleep(2)
                else:
                    yield (
                        "event: error\n"
                        f"data: {json.dumps({'error': 'Redis connection failed', 'message': 'Please ensure Redis is running'})}\n\n"
                    )
                    time.sleep(5)
            except Exception as e:
                error_count += 1
                yield (
                    "event: error\n"
                    f"data: {json.dumps({'error': str(e), 'message': 'An error occurred'})}\n\n"
                )
                time.sleep(2)
                if error_count > max_errors:
                    break

    response = Response(generate(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Get Prometheus metrics used by the frontend."""
    try:
        throughput = prom_instant("throughput_msgs_per_sec")
        latency = prom_instant("spark_processing_latency_ms")
        infer_rate = prom_instant("rate(model_inference_latency_ms_count[5s])")
        acc_tp = prom_instant("detection_true_positive_total")
        acc_fp = prom_instant("detection_false_positive_total")
        acc_fn = prom_instant("detection_false_negative_total")

        precision = (acc_tp / (acc_tp + acc_fp)) if (acc_tp + acc_fp) > 0 else 0.0
        recall = (acc_tp / (acc_tp + acc_fn)) if (acc_tp + acc_fn) > 0 else 0.0
        total_preds = acc_tp + acc_fp + acc_fn
        accuracy = (acc_tp / total_preds) if total_preds > 0 else 0.0

        return jsonify(
            {
                "throughput": round(throughput, 2),
                "latency": round(latency, 1),
                "inference_rate": round(infer_rate, 2),
                "precision": round(precision, 2),
                "recall": round(recall, 2),
                "accuracy": round(accuracy, 2),
                "true_positives": int(acc_tp),
                "false_positives": int(acc_fp),
                "false_negatives": int(acc_fn),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/transaction/<transaction_id>", methods=["GET"])
def get_transaction(transaction_id: str):
    """Get specific transaction by ID from Redis hash."""
    try:
        key = f"txn:{transaction_id}"
        data = redis_client.hgetall(key)
        if not data:
            return jsonify({"error": "Transaction not found"}), 404

        parsed: Dict[str, Any] = {}
        for k, v in data.items():
            try:
                parsed[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                parsed[k] = v

        return jsonify(parsed)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting backend API server...")
    print(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
    print(f"Prometheus: {PROMETHEUS_URL}")
    # Use host 0.0.0.0 so Docker/other hosts can reach it if needed
    app.run(host="0.0.0.0", port=5000, debug=True)


