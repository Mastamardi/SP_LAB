# Web Application Guide

## Overview

This project now includes a single-page web application (`index.html`) that replaces the Streamlit dashboard with a modern HTML/CSS/JavaScript frontend connected to a Flask backend API.

## Architecture

```
Frontend (index.html)
    ↓ HTTP/REST API + Server-Sent Events
Backend API (backend_api/server.py)
    ↓ Redis + Prometheus
Data Sources (Redis, Prometheus)
```

## Quick Start

### 1. Start Infrastructure
```bash
docker compose up -d
```

### 2. Start Backend Services
- Start Kafka Producer (streams data)
- Start Spark Streaming Job (processes transactions)
- **Start Backend API Server:**
  ```bash
  python backend_api/server.py
  ```
  Or use the convenience script:
  ```bash
  ./start_web_app.sh
  ```

### 3. Open Frontend

**Option 1: Direct file open**
- Simply open `index.html` in your web browser

**Option 2: Local web server (recommended)**
```bash
python -m http.server 8080
```
Then navigate to `http://localhost:8080` in your browser

## Features

### Home Page
- **Team Information**: Displays project guidance and team member details
- **Project Flow**: Explains the data pipeline from input to output
- **Problem Statement**: Describes the challenges addressed
- **Objectives**: Lists system goals
- **Methodology**: Details the technical approach
- **Conclusion**: Summarizes the system's value

### Live Processing Page
- **Real-time Transaction Stream**: Displays transactions as they are processed
- **ML Predictions**: Shows Normal (green) or Faulty (red) transaction indicators
- **Live Metrics Dashboard**: 
  - Throughput (msgs/s)
  - Processing Latency (ms)
  - Inference Rate (/s)
  - Precision
  - Recall
- **Connection Status**: Visual indicator of backend connectivity
- **Transaction Details**: Shows transaction ID, customer info, amounts, ML scores, etc.

## Backend API Endpoints

### Health Check
- `GET /api/health` - Check backend and Redis connection

### Predictions
- `GET /api/predictions/latest?limit=100` - Get latest predictions
- `GET /api/predictions/stream` - Server-Sent Events stream for real-time updates

### Metrics
- `GET /api/metrics` - Get Prometheus metrics (throughput, latency, precision, recall)

### Transaction Details
- `GET /api/transaction/<transaction_id>` - Get specific transaction

## Configuration

### Backend URL
To change the backend API URL, edit the `API_BASE_URL` constant in `index.html`:
```javascript
const API_BASE_URL = 'http://your-server:5000/api';
```

Or set it via `window.API_BASE_URL` before the script loads.

### Backend Environment Variables
The backend API server supports these environment variables:
- `REDIS_HOST` (default: `localhost`)
- `REDIS_PORT` (default: `6379`)
- `PROMETHEUS_URL` (default: `http://localhost:9090`)

Example:
```bash
REDIS_HOST=localhost REDIS_PORT=6379 python backend_api/server.py
```

## Troubleshooting

### Frontend can't connect to backend
1. Ensure the backend API server is running: `python backend_api/server.py`
2. Check that the API_BASE_URL in `index.html` matches your backend URL
3. Verify Redis is accessible (backend will show connection status)

### No transactions appearing
1. Ensure Kafka Producer is running and streaming data
2. Ensure Spark Streaming Job is running and processing transactions
3. Check Redis has data: `redis-cli LLEN predictions:latest`

### Metrics showing zeros
1. Ensure Prometheus is running and scraping Spark metrics
2. Verify Spark job is exposing metrics on port 8000
3. Check Prometheus URL configuration in backend

## Development

### Code Structure
- **Frontend**: Single-file application (`index.html`) with embedded CSS and JavaScript
- **Backend**: Flask REST API (`backend_api/server.py`)
- **Modular Design**: Code is organized for easy extension

### Future Enhancements
The codebase is designed to easily add:
- Real-time charts and graphs (using Chart.js or similar)
- Authentication and user management
- Additional ML model integrations
- Advanced filtering and search
- Export functionality

## Browser Compatibility

The application uses modern web APIs:
- Fetch API (for REST calls)
- Server-Sent Events (for streaming)
- CSS Grid and Flexbox (for layout)

Compatible with:
- Chrome/Edge (recommended)
- Firefox
- Safari
- Modern mobile browsers

## Security Notes

- CORS is enabled on the backend for development. Restrict in production.
- No authentication is implemented. Add authentication for production use.
- Backend runs on HTTP. Use HTTPS in production.
































