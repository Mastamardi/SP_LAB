# Backend API Server

This Flask-based API server exposes Redis data and Prometheus metrics for the frontend web application.

## Setup

1. Install dependencies:
```bash
pip install flask flask-cors redis requests
```

Or use the main requirements.txt:
```bash
pip install -r ../requirements.txt
```

## Configuration

The server uses environment variables for configuration (with defaults):
- `REDIS_HOST` (default: `localhost`)
- `REDIS_PORT` (default: `6379`)
- `PROMETHEUS_URL` (default: `http://localhost:9090`)

## Running the Server

```bash
python backend_api/server.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### Health Check
- `GET /api/health` - Check backend and Redis connection status

### Predictions
- `GET /api/predictions/latest?limit=100` - Get latest predictions (default limit: 100)
- `GET /api/predictions/stream` - Server-Sent Events stream for real-time predictions

### Metrics
- `GET /api/metrics` - Get Prometheus metrics (throughput, latency, precision, recall, etc.)

### Transaction Details
- `GET /api/transaction/<transaction_id>` - Get specific transaction by ID

## Usage with Frontend

1. Ensure Redis and Prometheus are running (via docker-compose)
2. Start the Spark streaming job (to populate Redis with predictions)
3. Start this API server: `python backend_api/server.py`
4. Open `index.html` in a web browser or serve it via a web server

## CORS

CORS is enabled to allow the frontend to connect from any origin. For production, restrict this appropriately.
































