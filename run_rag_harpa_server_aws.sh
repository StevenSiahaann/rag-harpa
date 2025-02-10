#!/bin/bash

SERVICE_NAME=rag-harpa

APP_DIR="/home/ubuntu/harpa/rag-harpa"
APP_FILE="app.py"
VENV_DIR="/home/ubuntu/harpa/harpa-env"
PYTHON_EXEC="$VENV_DIR/bin/python3"
GUNICORN_EXEC="$VENV_DIR/bin/gunicorn"

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment tidak ditemukan. Pastikan telah menginstallnya di $VENV_DIR"
    exit 1
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel gunicorn


SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME.service"
echo "[Unit]
Description=RAG Harpa Flask Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=$APP_DIR
ExecStart=$GUNICORN_EXEC -w 4 -b 0.0.0.0:8000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
" | sudo tee $SERVICE_FILE > /dev/null

sudo systemctl daemon-reload

sudo systemctl enable $SERVICE_NAME

sudo systemctl start $SERVICE_NAME

sudo systemctl status $SERVICE_NAME
