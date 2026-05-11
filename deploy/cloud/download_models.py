#!/usr/bin/env python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import Blip2Processor, Blip2ForConditionalGeneration

models = [
    "Salesforce/blip2-opt-6.7b-coco",
    "Salesforce/blip2-opt-6.7b",
    "Salesforce/blip2-flan-t5-xxl",
    "Salesforce/blip2-flan-t5-xl",
]

for model_name in models:
    print(f"\n{'='*50}")
    print(f"Downloading: {model_name}")
    print('='*50)
    try:
        print("Downloading processor...")
        processor = Blip2Processor.from_pretrained(model_name, force_download=True)
        print("Downloading model...")
        model = Blip2ForConditionalGeneration.from_pretrained(model_name, force_download=True)
        print(f"SUCCESS: {model_name} downloaded!")
    except Exception as e:
        print(f"ERROR downloading {model_name}: {e}")

print("\nAll downloads completed!")