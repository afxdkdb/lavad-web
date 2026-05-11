import os
import sys
import json
import uuid
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import cv2
from tqdm import tqdm


class VideoAnalyzer:
    def __init__(
        self,
        blip2_model_path: str = None,
        qwen_model_path: str = None,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.blip2_model_path = blip2_model_path or os.environ.get("BLIP2_MODEL_PATH")
        self.qwen_model_path = qwen_model_path or os.environ.get("QWEN_MODEL_PATH")
        self.temp_dir = Path(tempfile.mkdtemp())
        self.blip2_model = None
        self.qwen_model = None
        self.blip2_processor = None
        self.imagebind_model = None
        self._models_loaded = False

    def _load_blip2(self):
        if self.blip2_model is not None:
            return
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        print(f"Loading BLIP-2 model from {self.blip2_model_path}...")
        self.blip2_processor = Blip2Processor.from_pretrained(self.blip2_model_path)
        self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
            self.blip2_model_path,
            torch_dtype=torch.float16
        )
        self.blip2_model.to(self.device)
        self.blip2_model.eval()
        print("BLIP-2 model loaded successfully.")

    def _load_qwen(self):
        if self.qwen_model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading Qwen-7B model from {self.qwen_model_path}...")
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            self.qwen_model_path,
            trust_remote_code=True
        )
        self.qwen_tokenizer.chat_template = """{% for message in messages %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}<|im_start|>assistant"""
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            self.qwen_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.qwen_model.eval()
        print("Qwen-7B model loaded successfully.")

    def _load_imagebind(self):
        if self.imagebind_model is not None:
            return
        sys.path.append(str(Path(__file__).parent.parent.parent.parent / "libs" / "ImageBind"))
        from imagebind.models.imagebind_model import imagebind_huge
        from imagebind.models.helpers import clamp_exp
        print("Loading ImageBind model...")
        self.imagebind_model = imagebind_huge(pretrained=True)
        self.imagebind_model.to(self.device)
        self.imagebind_model.eval()
        print("ImageBind model loaded successfully.")

    def load_models(self):
        if self._models_loaded:
            return
        self._load_imagebind()
        self._load_blip2()
        self._load_qwen()
        self._models_loaded = True

    def extract_frames(self, video_path: str, frame_interval: int = 16) -> Tuple[List[str], float, int]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        frames = []
        frame_idx = 0
        saved_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                frame_path = self.temp_dir / f"frame_{saved_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frames.append(str(frame_path))
                saved_idx += 1
            frame_idx += 1

        cap.release()
        return frames, fps, saved_idx

    def generate_caption(self, frame_path: str) -> str:
        import torch
        from PIL import Image

        image = Image.open(frame_path).convert("RGB")
        inputs = self.blip2_processor(images=image, return_tensors="pt").to(
            self.device, dtype=torch.float16
        )
        generated_ids = self.blip2_model.generate(**inputs)
        caption = self.blip2_processor.decode(generated_ids[0], skip_special_tokens=True)
        return caption.strip()

    def generate_captions_batch(self, frame_paths: List[str]) -> List[str]:
        from PIL import Image
        import torch

        images = [Image.open(fp).convert("RGB") for fp in frame_paths]
        inputs = self.blip2_processor(images=images, return_tensors="pt").to(
            self.device, dtype=torch.float16
        )
        generated_ids = self.blip2_model.generate(**inputs)
        captions = [
            self.blip2_processor.decode(gid, skip_special_tokens=True).strip()
            for gid in generated_ids
        ]
        return captions

    def query_qwen(self, prompt: str, max_gen_len: int = 256, temperature: float = 0.6) -> str:
        import torch
        dialog = [
            {"role": "system", "content": "You are a security surveillance analyst analyzing video content."},
            {"role": "user", "content": prompt}
        ]
        prompt_text = self.qwen_tokenizer.apply_chat_template(
            dialog,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.qwen_tokenizer(prompt_text, return_tensors="pt").to(self.qwen_model.device)

        with torch.no_grad():
            outputs = self.qwen_model.generate(
                **inputs,
                max_new_tokens=max_gen_len,
                temperature=temperature,
                top_p=0.9,
                do_sample=temperature > 0
            )

        response = self.qwen_tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()

    def score_anomaly_llm(self, captions: List[str], frame_idxs: List[int]) -> List[float]:
        if len(captions) == 0:
            return []

        context_prompt = """You are analyzing a video sequence for anomalies. Given the following frame descriptions, rate the likelihood of anomalous activity on a scale from 0 to 10, where 0 is completely normal and 10 is highly anomalous.

Important: Consider only truly suspicious events like:
- Violence or fighting
- Theft or burglary
- Vandalism
- Suspicious behavior
- Accidents

Return your answer in the format: [SCORE] (e.g., [5.5])"""

        format_prompt = """
Describe briefly what happens and rate the anomaly level: [SCORE]"""

        frame_descriptions = "\n".join([
            f"Frame {idx}: {cap}" for idx, cap in zip(frame_idxs, captions)
        ])

        prompt = f"""Analyze this video segment:

{frame_descriptions}

{format_prompt}"""

        try:
            response = self.query_qwen(prompt, max_gen_len=128, temperature=0.3)
            import re
            pattern = r"\[?(\d+(?:\.\d+)?)\]?"
            match = re.search(pattern, response)
            if match:
                score = float(match.group(1))
                score = max(0, min(10, score))
                return [score / 10.0] * len(captions)
        except Exception as e:
            print(f"Error scoring with LLM: {e}")

        return [0.5] * len(captions)

    def analyze_video(self, video_path: str) -> Dict:
        start_time = time.time()
        video_id = str(uuid.uuid4())
        video_name = Path(video_path).name

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        frames, video_fps, sampled_frames = self.extract_frames(video_path, frame_interval=16)

        captions = []
        all_scores = []
        top_anomaly_captions = []

        for i in tqdm(range(0, len(frames), 8), desc="Analyzing frames"):
            batch_frames = frames[i:i+8]
            batch_captions = self.generate_captions_batch(batch_frames)
            captions.extend(batch_captions)

            batch_scores = self.score_anomaly_llm(
                batch_captions,
                list(range(i, min(i+8, len(frames))))
            )
            all_scores.extend(batch_scores)

        threshold = np.mean(all_scores) + np.std(all_scores)
        threshold = max(threshold, 0.5)

        anomaly_frames = []
        for idx, (score, caption) in enumerate(zip(all_scores, captions)):
            if score > threshold:
                timestamp = idx * 16 / video_fps if video_fps > 0 else idx * 16 / 30
                anomaly_frames.append({
                    "frame_idx": idx,
                    "timestamp": round(timestamp, 2),
                    "score": round(float(score), 4),
                    "caption": caption
                })
                if len(top_anomaly_captions) < 5:
                    top_anomaly_captions.append(caption)

        anomaly_ratio = len(anomaly_frames) / len(frames) if len(frames) > 0 else 0
        overall_score = float(np.mean(all_scores)) if all_scores else 0.0

        summary = self._generate_summary(anomaly_frames, top_anomaly_captions, anomaly_ratio)

        processing_time = time.time() - start_time

        for f in frames:
            try:
                os.remove(f)
            except:
                pass

        return {
            "video_id": video_id,
            "video_name": video_name,
            "total_frames": sampled_frames,
            "fps": round(video_fps, 2),
            "duration": round(duration, 2),
            "anomaly_frames": anomaly_frames,
            "normal_frames": sampled_frames - len(anomaly_frames),
            "abnormal_frames": len(anomaly_frames),
            "anomaly_ratio": round(anomaly_ratio, 4),
            "overall_score": round(overall_score, 4),
            "summary": summary,
            "top_anomaly_captions": top_anomaly_captions,
            "processing_time": round(processing_time, 2)
        }

    def _generate_summary(self, anomaly_frames: List, top_captions: List, anomaly_ratio: float) -> str:
        if len(anomaly_frames) == 0:
            return "No anomalous events detected in the video. The video appears to show normal activity."

        top_score = max([f["score"] for f in anomaly_frames]) if anomaly_frames else 0
        avg_score = np.mean([f["score"] for f in anomaly_frames]) if anomaly_frames else 0

        summary_parts = []
        summary_parts.append(f"Detected {len(anomaly_frames)} potentially anomalous frame(s) ({(anomaly_ratio*100):.1f}% of video).")

        if anomaly_ratio < 0.1:
            summary_parts.append("Most of the video appears normal with brief anomalous moments.")
        elif anomaly_ratio < 0.3:
            summary_parts.append("Several portions of the video show potentially suspicious activity.")
        else:
            summary_parts.append("A significant portion of the video contains anomalous events.")

        if top_score > 0.7:
            summary_parts.append("High-confidence anomaly detected. Immediate attention may be required.")
        elif top_score > 0.5:
            summary_parts.append("Moderate anomaly levels detected. Review is recommended.")

        return " ".join(summary_parts)

    def get_model_status(self) -> Dict:
        return {
            "blip2_loaded": self.blip2_model is not None,
            "qwen_loaded": self.qwen_model is not None,
            "imagebind_loaded": self.imagebind_model is not None
        }

    def cleanup(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)