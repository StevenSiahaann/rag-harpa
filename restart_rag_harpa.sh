#!/bin/bash

echo "Stopping existing Flask process..."
pkill -9 -f "python app.py"

sleep 2

echo "Starting Flask application..."
./run_rag_harpa.sh

echo "Flask restarted successfully."
