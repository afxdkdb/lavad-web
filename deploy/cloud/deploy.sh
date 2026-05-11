#!/bin/bash

set -e

echo "=========================================="
echo "LAVAD Web System - Cloud Deployment Script"
echo "=========================================="

SERVER_IP="121.48.164.7"
SERVER_PORT="54291"
SERVER_USER="jinanyang"
PROJECT_PATH="/home/jinanyang/lavad"

echo "Connecting to server: ${SERVER_USER}@${SERVER_IP}:${SERVER_PORT}"

echo "Step 1: Syncing project files..."
rsync -avz -e "ssh -p ${SERVER_PORT}" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='node_modules' \
    --exclude='.venv' \
    --exclude='venv' \
    --exclude='*.log' \
    --exclude='nohup.out' \
    ./ ${SERVER_USER}@${SERVER_IP}:${PROJECT_PATH}/

echo "Step 2: Installing dependencies on server..."
ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_IP} << 'ENDSSH'
    cd /home/jinanyang/lavad

    source /home/jinanyang/miniconda/etc/profile.d/conda.sh
    conda activate lavad

    pip install fastapi uvicorn python-multipart streamlit requests urllib3 aiofiles -i https://pypi.tuna.tsinghua.edu.cn/simple

    echo "Dependencies installed successfully!"
ENDSSH

echo "Step 3: Starting backend service..."
ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_IP} << 'ENDSSH'
    cd /home/jinanyang/lavad

    source /home/jinanyang/miniconda/etc/profile.d/conda.sh
    conda activate lavad

    export PYTHONPATH=/home/jinanyang/lavad:/home/jinanyang/lavad/libs/ImageBind:$PYTHONPATH
    export HF_ENDPOINT=https://hf-mirror.com
    export BLIP2_MODEL_PATH=/home/jinanyang/lavad/libs/blip2-opt-6.7b-coco
    export QWEN_MODEL_PATH=/home/jinanyang/lavad/libs/Qwen-7B-Chat

    pkill -f uvicorn || true
    nohup python -m uvicorn deploy.service.backend.main:app --host 0.0.0.0 --port 8000 --workers 1 > /tmp/lavad_backend.log 2>&1 &

    sleep 3

    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Backend started successfully!"
    else
        echo "Warning: Backend may not have started correctly. Check logs."
        cat /tmp/lavad_backend.log
    fi
ENDSSH

echo "Step 4: Starting frontend service..."
ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_IP} << 'ENDSSH'
    cd /home/jinanyang/lavad

    pkill -f streamlit || true
    nohup streamlit run deploy/service/frontend/app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true > /tmp/lavad_frontend.log 2>&1 &

    sleep 3

    if curl -s http://localhost:8501 > /dev/null 2>&1; then
        echo "Frontend started successfully!"
    else
        echo "Warning: Frontend may not have started correctly. Check logs."
        cat /tmp/lavad_frontend.log
    fi
ENDSSH

echo ""
echo "=========================================="
echo "Deployment completed!"
echo "Access the web interface at: http://${SERVER_IP}:8501"
echo "API documentation at: http://${SERVER_IP}:8000/docs"
echo "=========================================="