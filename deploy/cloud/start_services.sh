#!/bin/bash
cd /home/jinanyang/lavad_new
source /home/jinanyang/miniconda/etc/profile.d/conda.sh
conda activate lavad

export PYTHONPATH=/home/jinanyang/lavad_new:/home/jinanyang/lavad_new/libs/ImageBind:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export BLIP2_MODEL_PATH=/home/jinanyang/lavad/libs/blip2-opt-6.7b-coco
export QWEN_MODEL_PATH=/home/jinanyang/lavad/libs/Qwen-7B-Chat

echo "=========================================="
echo "LAVAD Video Anomaly Detection System"
echo "=========================================="
echo ""
echo "Stopping old services..."
pkill -f 'uvicorn.*8000' || true
pkill -f 'streamlit.*8501' || true
sleep 2

echo "Starting backend service (FastAPI) on port 8000..."
nohup python -m uvicorn deploy.service.backend.main:app --host 0.0.0.0 --port 8000 --workers 1 > /tmp/lavad_backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

sleep 3

echo "Starting frontend service (Streamlit) on port 8501..."
nohup streamlit run deploy/service/frontend/app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true > /tmp/lavad_frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID"

sleep 3

echo ""
echo "=========================================="
echo "Services Started Successfully!"
echo "=========================================="
echo "Web Interface: http://121.48.164.7:8501"
echo "API Docs:      http://121.48.164.7:8000/docs"
echo ""
echo "Log files:"
echo "  Backend:  /tmp/lavad_backend.log"
echo "  Frontend: /tmp/lavad_frontend.log"
echo "=========================================="