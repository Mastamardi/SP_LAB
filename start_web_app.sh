#!/bin/bash

# Startup script for the Web Application
# This script starts the backend API server

echo "=========================================="
echo "Starting Fraud Detection Web Application"
echo "=========================================="
echo ""

# Check if Redis is accessible
echo "Checking Redis connection..."
if command -v redis-cli &> /dev/null; then
    if redis-cli -h localhost -p 6379 ping &> /dev/null; then
        echo "✓ Redis is accessible"
    else
        echo "✗ Redis is not accessible. Make sure Docker services are running:"
        echo "  docker compose up -d"
        exit 1
    fi
else
    echo "⚠ redis-cli not found, skipping Redis check"
fi

echo ""
echo "Starting Backend API Server..."
echo "API will be available at: http://localhost:5000"
echo ""
echo "To open the frontend:"
echo "  1. Open index.html in your browser, OR"
echo "  2. Run: python -m http.server 8080"
echo "     Then open: http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the Flask server
python backend_api/server.py
































