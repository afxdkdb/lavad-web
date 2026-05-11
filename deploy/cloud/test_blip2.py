import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['BLIP2_MODEL_PATH'] = '/home/jinanyang/lavad/libs/blip2-opt-6.7b-coco'

from transformers import Blip2ForConditionalGeneration, Blip2Processor

print(f"Loading BLIP-2 from {os.environ['BLIP2_MODEL_PATH']}...")
try:
    processor = Blip2Processor.from_pretrained(os.environ['BLIP2_MODEL_PATH'])
    print("Processor loaded successfully!")
    model = Blip2ForConditionalGeneration.from_pretrained(
        os.environ['BLIP2_MODEL_PATH'],
        torch_dtype=torch.float16
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
