#!/bin/bash
cd /home/jinanyang/lavad_new
source /home/jinanyang/miniconda/etc/profile.d/conda.sh
conda activate lavad

export PYTHONPATH=/home/jinanyang/lavad_new:/home/jinanyang/lavad/libs/ImageBind:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export BLIP2_MODEL_PATH=/home/jinanyang/lavad/libs/blip2-opt-6.7b-coco
export QWEN_MODEL_PATH=/home/jinanyang/lavad/libs/Qwen-7B-Chat

pkill -9 -f uvicorn || true
sleep 2

echo "Starting backend with full 7-step LAVAD pipeline..."
nohup python -m uvicorn deploy.service.backend.main:app --host 0.0.0.0 --port 8000 --workers 1 > /tmp/lavad_backend.log 2>&1 &
echo "Backend started. PID: $!"