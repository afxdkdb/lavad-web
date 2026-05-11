#!/bin/bash
cd /home/jinanyang/lavad_new
source /home/jinanyang/miniconda/etc/profile.d/conda.sh
conda activate lavad

export CUDA_VISIBLE_DEVICES=0,1,2
export PYTHONPATH=/home/jinanyang/lavad_new:/home/jinanyang/lavad/libs/ImageBind
export HF_ENDPOINT=https://hf-mirror.com
export QWEN_MODEL_PATH=/data/jinanyang/models/Qwen-7B-Chat

echo "Starting backend with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nohup python -m uvicorn deploy.service.backend.main:app --host 0.0.0.0 --port 8000 --workers 1 > /tmp/lavad_backend.log 2>&1 &
echo "Backend started. PID: $!"