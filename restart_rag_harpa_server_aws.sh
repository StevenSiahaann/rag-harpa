#!/bin/bash

SERVICE_NAME=rag-harpa

sudo systemctl stop $SERVICE_NAME

sudo systemctl daemon-reload

VENV_DIR="/home/ubuntu/harpa/harpa-env"
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment tidak ditemukan. Pastikan telah menginstallnya di $VENV_DIR"
    exit 1
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel gunicorn

sudo systemctl restart $SERVICE_NAME

sudo systemctl status $SERVICE_NAME