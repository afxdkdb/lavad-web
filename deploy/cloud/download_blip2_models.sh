#!/bin/bash
source /home/jinanyang/miniconda/etc/profile.d/conda.sh
conda activate lavad

export HF_ENDPOINT=https://hf-mirror.com

models=(
    "Salesforce/blip2-opt-6.7b-coco"
    "Salesforce/blip2-opt-6.7b"
    "Salesforce/blip2-flan-t5-xxl"
    "Salesforce/blip2-flan-t5-xl"
)

echo "Downloading BLIP-2 models..."
for model in "${models[@]}"; do
    echo "=========================================="
    echo "Downloading: $model"
    echo "=========================================="
    python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading $model...')
try:
    tokenizer = AutoTokenizer.from_pretrained('$model', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained('$model')
    print('Downloaded successfully: $model')
except Exception as e:
    print(f'Error downloading $model: {e}')
"
done
echo "Done!"