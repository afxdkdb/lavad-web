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
import re
import faiss

sys.path.insert(0, "/home/jinanyang/lavad/libs/ImageBind")
os.chdir("/home/jinanyang/lavad_new")
from imagebind.models.imagebind_model import imagebind_huge, ModalityType
from imagebind import data

SUMMARY_PROMPT = "Please summarize what happened in few sentences, based on the following temporal description of a scene. Do not include any unnecessary details or descriptions."
CONTEXT_PROMPT = "If you are a law enforcement agency, how would you rate the scene described on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious activities?"
FORMAT_PROMPT = "Please provide the response in the form of a Python list and respond with only one number in the provided list below [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] without any textual explanation. It should begin with '[' and end with ']'."

BLIP2_MODELS = [
    "/data/jinanyang/models/blip2-flan-t5-xl",
    "/data/jinanyang/models/blip2-flan-t5-xl-coco",
    "/data/jinanyang/models/blip2-flan-t5-xxl",
    "/data/jinanyang/models/blip2-opt-6.7b",
    "/data/jinanyang/models/blip2-opt-6.7b-coco",
]

FRAME_INTERVAL = 16
FRAME_INTERVAL_SHORT = 4
FRAME_INTERVAL_MEDIUM = 8
MIN_CLIPS_FOR_REFINEMENT = 20
MIN_CLIPS_FOR_STATS = 10
BATCH_SIZE = 4
CLIP_DURATION = 10
NUM_SAMPLES = 10
NUM_NEIGHBORS_STEP3 = 1
NUM_NEIGHBORS_STEP6 = 10
FPS = 30
INDEX_DIM = 1024


def uniform_temporal_subsample(frame_paths: List[str], num_samples: int) -> List[str]:
    if len(frame_paths) <= num_samples:
        return frame_paths
    indices = np.linspace(0, len(frame_paths) - 1, num_samples, dtype=int)
    return [frame_paths[i] for i in indices]


class LAVADPipeline:
    def __init__(
        self,
        qwen_model_path: str = None,
        device: str = None
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # 优先使用环境变量或传入的路径，否则使用默认路径
        if qwen_model_path:
            self.qwen_model_path = qwen_model_path
        elif os.environ.get("QWEN_MODEL_PATH"):
            self.qwen_model_path = os.environ.get("QWEN_MODEL_PATH")
        else:
            # 尝试多个可能的默认路径
            possible_paths = [
                "/data/jinanyang/models/Qwen-7B-Chat",  # 云端部署路径
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "libs", "Qwen-7B-Chat"),
                os.path.join(os.path.expanduser("~"), "lavad", "libs", "Qwen-7B-Chat"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    self.qwen_model_path = path
                    break
            else:
                self.qwen_model_path = possible_paths[0]
        self.temp_dir = Path(tempfile.mkdtemp())
        self.loaded_blip2_models = {}
        self.qwen_model = None
        self.qwen_tokenizer = None
        self.imagebind_model = None
        self._imagebind_on_cpu = False
        self._qwen_on_cpu = False

    def _load_single_blip2(self, model_path: str):
        if model_path in self.loaded_blip2_models:
            return self.loaded_blip2_models[model_path]

        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        print(f"Loading BLIP-2 model from {model_path}...")
        model_name = os.path.basename(model_path)

        try:
            processor = Blip2Processor.from_pretrained(model_path, local_files_only=True)
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=True
            )
            model.eval()
            self.loaded_blip2_models[model_path] = {
                "processor": processor,
                "model": model
            }
            print(f"  {model_name} loaded successfully.")
            return self.loaded_blip2_models[model_path]
        except Exception as e:
            print(f"  Failed to load {model_name}: {e}")
            return None

    def _unload_blip2(self, model_path: str):
        if model_path in self.loaded_blip2_models:
            model = self.loaded_blip2_models[model_path]["model"]
            model.to("cpu")
            del model
            del self.loaded_blip2_models[model_path]
            torch.cuda.empty_cache()
            print(f"  Unloaded {os.path.basename(model_path)}")

    def _load_qwen(self):
        if self.qwen_model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading Qwen-7B model from {self.qwen_model_path}...")
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            self.qwen_model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        self.qwen_tokenizer.chat_template = """{% for message in messages %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|}>
{% endfor %}<|im_start|>assistant"""
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            self.qwen_model_path,
            torch_dtype=torch.float16,
            device_map="cuda:1",
            trust_remote_code=True,
            local_files_only=True
        )
        self.qwen_model.eval()
        print("Qwen-7B model loaded successfully.")

    def _load_imagebind(self):
        if self.imagebind_model is not None:
            return
        print("Loading ImageBind model...")
        self.imagebind_model = imagebind_huge(pretrained=True)
        self.imagebind_model.to("cuda:0")
        self.imagebind_model.eval()
        print("ImageBind model loaded successfully.")

    def _temp_unload_for_blip2(self):
        if self.imagebind_model is not None and not self._imagebind_on_cpu:
            del self.imagebind_model
            self.imagebind_model = None
            torch.cuda.empty_cache()
            self._imagebind_on_cpu = True
            print("  Temporarily unloaded ImageBind to CPU")

        if self.qwen_model is not None and not self._qwen_on_cpu:
            del self.qwen_model
            self.qwen_model = None
            torch.cuda.empty_cache()
            self._qwen_on_cpu = True
            print("  Temporarily unloaded Qwen to CPU")

    def _restore_after_blip2(self):
        if self._imagebind_on_cpu and self.imagebind_model is None:
            print("  Restoring ImageBind to GPU...")
            self.imagebind_model = imagebind_huge(pretrained=True)
            self.imagebind_model.to("cuda:0")
            self.imagebind_model.eval()
            self._imagebind_on_cpu = False
            torch.cuda.empty_cache()
            print("  ImageBind restored")

        if self._qwen_on_cpu and self.qwen_model is None:
            print("  Restoring Qwen to GPU...")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                self.qwen_model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            self.qwen_tokenizer.chat_template = """{% for message in messages %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|}>
{% endfor %}<|im_start|>assistant"""
            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                self.qwen_model_path,
                torch_dtype=torch.float16,
                device_map="cuda:1",
                trust_remote_code=True,
                local_files_only=True
            )
            self.qwen_model.eval()
            self._qwen_on_cpu = False
            torch.cuda.empty_cache()
            print("  Qwen restored")

    def load_models(self):
        self._load_imagebind()
        self._load_qwen()

    def get_model_status(self) -> Dict:
        return {
            "blip2_loaded": list(self.loaded_blip2_models.keys()),
            "blip2_available": BLIP2_MODELS,
            "qwen_loaded": self.qwen_model is not None,
            "imagebind_loaded": self.imagebind_model is not None
        }

    def step0_extract_frames(self, video_path: str, frame_interval: int = None) -> Tuple[List[str], float, int]:
        print("\n" + "="*60)
        print("STEP 0: Extracting video frames...")
        print("="*60)

        interval = frame_interval or FRAME_INTERVAL

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
            if frame_idx % interval == 0:
                frame_path = self.temp_dir / f"frame_{saved_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frames.append(str(frame_path))
                saved_idx += 1
            frame_idx += 1

        cap.release()
        print(f"Extracted {len(frames)} frames at interval={interval} (FPS: {fps:.2f}, Duration: {duration:.2f}s)")
        return frames, fps, saved_idx

    def step1_generate_captions(self, frame_paths: List[str]) -> Dict[str, List[str]]:
        print("\n" + "="*60)
        print("STEP 1: Generating captions with 5 BLIP-2 models...")
        print("="*60)

        from PIL import Image

        all_captions = {model: [""] * len(frame_paths) for model in BLIP2_MODELS}

        self._temp_unload_for_blip2()

        for model_idx, model_path in enumerate(BLIP2_MODELS):
            model_name = os.path.basename(model_path)
            print(f"\n[{model_idx+1}/5] Processing with {model_name}...")

            blip2_data = self._load_single_blip2(model_path)
            if blip2_data is None:
                print(f"  Skipping {model_name} (failed to load)")
                self._unload_blip2(model_path)
                continue

            processor = blip2_data["processor"]
            model = blip2_data["model"]

            try:
                for i in tqdm(range(0, len(frame_paths), BATCH_SIZE), desc=f"  {model_name}"):
                    batch_frames = frame_paths[i:i+BATCH_SIZE]
                    images = [Image.open(fp).convert("RGB") for fp in batch_frames]
                    inputs = processor(images=images, return_tensors="pt").to(model.device)

                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, max_new_tokens=50)

                    captions = [processor.decode(gid, skip_special_tokens=True).strip() for gid in generated_ids]

                    for j, caption in enumerate(captions):
                        all_captions[model_path][i + j] = caption

                print(f"  {model_name}: Generated {len(frame_paths)} captions")

            except Exception as e:
                print(f"  Error with {model_name}: {e}")

            self._unload_blip2(model_path)

        self._restore_after_blip2()

        return all_captions

    def step2_create_text_index(self, captions_dict: Dict[str, List[str]]) -> Tuple[faiss.Index, List[str]]:
        print("\n" + "="*60)
        print("STEP 2: Creating ImageBind text index (with deduplication)...")
        print("="*60)

        num_clips = len(captions_dict[BLIP2_MODELS[0]])

        caption_to_frame_idxs = {}
        for frame_idx in range(num_clips):
            frame_unique_captions = set()
            for model_path in BLIP2_MODELS:
                caption = captions_dict[model_path][frame_idx]
                if caption:
                    frame_unique_captions.add(caption)
            for caption in frame_unique_captions:
                if caption not in caption_to_frame_idxs:
                    caption_to_frame_idxs[caption] = []
                caption_to_frame_idxs[caption].append(frame_idx)

        all_texts = []
        file_names = []

        for frame_idx in range(num_clips):
            frame_unique_captions = set()
            for model_path in BLIP2_MODELS:
                caption = captions_dict[model_path][frame_idx]
                if caption:
                    frame_unique_captions.add(caption)

            unique_captions = []
            for caption in frame_unique_captions:
                valid_indices = [idx for idx in caption_to_frame_idxs[caption] if idx % FRAME_INTERVAL == 0]
                if valid_indices and frame_idx == min(valid_indices):
                    unique_captions.append(caption)

            for caption in unique_captions:
                model_idx = next(
                    i for i, model_path in enumerate(BLIP2_MODELS)
                    if captions_dict[model_path][frame_idx] == caption
                )
                all_texts.append(caption)
                file_names.append(f"{os.path.basename(BLIP2_MODELS[model_idx])}/video/{frame_idx}")

        print(f"Total unique text embeddings: {len(all_texts)} (deduplicated)")

        text_inputs = data.load_and_transform_text(all_texts, "cuda:0")

        with torch.no_grad():
            text_embeddings = self.imagebind_model({ModalityType.TEXT: text_inputs})
            text_vectors = text_embeddings[ModalityType.TEXT].cpu().numpy()

        faiss.normalize_L2(text_vectors)

        index = faiss.IndexFlatIP(INDEX_DIM)
        index.add(text_vectors)

        print(f"Text index created with {index.ntotal} vectors")
        return index, file_names

    def step3_clean_captions(self, captions_dict: Dict[str, List[str]], frame_paths: List[str], text_index: faiss.Index, file_names: List[str]) -> Dict[str, Dict]:
        print("\n" + "="*60)
        print(f"STEP 3: Cross-modal caption cleaning (k={NUM_NEIGHBORS_STEP3})...")
        print("="*60)

        frames_per_clip = int(CLIP_DURATION * FPS)
        num_clips = len(frame_paths)

        video_captions_retrieved = {model: {} for model in BLIP2_MODELS}

        print(f"Processing {num_clips} clips with {NUM_NEIGHBORS_STEP3} neighbors...")

        for clip_idx in tqdm(range(0, num_clips), desc="Cleaning captions"):
            clip_center_frame = clip_idx
            start_frame = max(clip_center_frame - frames_per_clip // 2, 0)
            end_frame = min(clip_center_frame + frames_per_clip // 2, len(frame_paths))
            clip_frame_paths = frame_paths[start_frame:end_frame]

            clip_subsample_paths = uniform_temporal_subsample(clip_frame_paths, NUM_SAMPLES)

            vision_inputs = data.load_and_transform_vision_data(clip_subsample_paths, "cuda:0")

            with torch.no_grad():
                vision_embeddings = self.imagebind_model({ModalityType.VISION: vision_inputs})
                vision_vectors = vision_embeddings[ModalityType.VISION].cpu().numpy()

            if vision_vectors.ndim == 1:
                vision_vectors = vision_vectors.reshape(1, -1)

            faiss.normalize_L2(vision_vectors)
            D, I = text_index.search(vision_vectors, NUM_NEIGHBORS_STEP3)

            seen_captions = {mp: set() for mp in BLIP2_MODELS}

            for frame_local_idx in range(vision_vectors.shape[0]):
                neighbor_idx = I[frame_local_idx][0]
                if 0 <= neighbor_idx < len(file_names):
                    file_name = file_names[neighbor_idx]
                    parts = file_name.split("/")
                    if len(parts) >= 3:
                        ret_cap_model_name = parts[0]
                        ret_frame_idx = parts[2]

                        try:
                            model_path = next(
                                mp for mp in BLIP2_MODELS
                                if os.path.basename(mp) == ret_cap_model_name
                            )
                        except StopIteration:
                            continue

                        ret_frame_int = int(ret_frame_idx)
                        if 0 <= ret_frame_int < len(captions_dict.get(model_path, [])):
                            caption = captions_dict[model_path][ret_frame_int]
                            if caption:
                                seen_captions[model_path].add(caption)

            for model_path in BLIP2_MODELS:
                captions_list = list(seen_captions[model_path])
                if captions_list:
                    video_captions_retrieved[model_path][str(clip_center_frame)] = {}
                    for ci, cap in enumerate(captions_list):
                        video_captions_retrieved[model_path][str(clip_center_frame)][str(ci)] = {
                            "caption": cap,
                            "similarity": 1.0
                        }
                else:
                    video_captions_retrieved[model_path][str(clip_center_frame)] = {}

        for model_path in BLIP2_MODELS:
            count = len([k for k, v in video_captions_retrieved[model_path].items() if v])
            print(f"  {os.path.basename(model_path)}: Cleaned {count} clips")

        return video_captions_retrieved

    def step4_llm_summaries_and_scores(self, cleaned_captions: Dict[str, Dict]) -> Tuple[List[str], List[float]]:
        print("\n" + "="*60)
        print("STEP 4: Generating summaries and scores with LLM...")
        print("="*60)

        self._load_qwen()

        num_clips = len(cleaned_captions[BLIP2_MODELS[0]])
        all_summaries = []
        all_scores = []

        for clip_idx in tqdm(range(num_clips), desc="LLM Processing"):
            clip_captions = []
            for model_path in BLIP2_MODELS:
                if str(clip_idx) in cleaned_captions[model_path]:
                    for nn_idx_str, nn_data in cleaned_captions[model_path][str(clip_idx)].items():
                        caption = nn_data["caption"]
                        if caption:
                            clip_captions.append(f"[{os.path.basename(model_path)}] {caption}")

            combined_caption = " ".join(clip_captions)

            summary = self._generate_summary(combined_caption, clip_idx)
            all_summaries.append(summary)

            score = self._score_summary(summary, clip_idx)
            all_scores.append(score)

        scores_array = np.array([score for score in all_scores if score != -1])
        if len(scores_array) > 0:
            all_scores = self._interpolate_unmatched_scores(all_scores)

        final_scores = np.array(all_scores)
        print(f"Generated {len(all_summaries)} summaries and scores")
        print(f"  Score stats: min={final_scores.min():.4f}, max={final_scores.max():.4f}, "
              f"mean={final_scores.mean():.4f}, std={final_scores.std():.4f}")
        return all_summaries, all_scores

    def _generate_summary(self, caption: str, clip_idx: int = 0) -> str:
        return self._query_qwen(caption, system_prompt=SUMMARY_PROMPT, max_gen_len=128, temperature=0.6)

    def _score_summary(self, summary: str, clip_idx: int = 0) -> float:
        scoring_prompt = f"{CONTEXT_PROMPT} {FORMAT_PROMPT}"
        response = self._query_qwen(summary, system_prompt=scoring_prompt, max_gen_len=64, temperature=0.6)
        score = self._parse_score(response)
        if clip_idx < 5:
            print(f"  [Clip {clip_idx}] Score: {score} (raw: {response})")
        return score

    def _query_qwen(self, user_content: str, system_prompt: str = None, max_gen_len: int = 256, temperature: float = 0.6) -> str:
        if self.qwen_model is None or self.qwen_tokenizer is None:
            return "0.5"

        try:
            dialog = [
                {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": user_content}
            ]
            prompt_text = self.qwen_tokenizer.apply_chat_template(
                dialog, tokenize=False, add_generation_prompt=True
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
        except Exception as e:
            print(f"Error querying Qwen: {e}")
            return "0.5"

    def _parse_score(self, response: str) -> float:
        pattern = r"\[(\d+(?:\.\d+)?)\]"
        match = re.search(pattern, response)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        fallback = r"(\d+\.\d+)"
        match = re.search(fallback, response)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        return -1

    def _interpolate_unmatched_scores(self, scores: List[float]) -> List[float]:
        indexed_scores = {i: s for i, s in enumerate(scores)}
        valid = [(i, s) for i, s in indexed_scores.items() if s != -1]
        if not valid:
            return [0.0] * len(scores)
        if len(valid) == 1:
            return [max(0.0, min(1.0, valid[0][1]))] * len(scores)
        all_indices = list(indexed_scores.keys())
        valid_indices, valid_values = zip(*valid)
        interpolated = np.interp(all_indices, valid_indices, valid_values)
        return [max(0.0, min(1.0, round(float(v), 4))) for v in interpolated]

    def step5_create_summary_index(self, summaries: List[str]) -> Tuple[faiss.Index, List[int]]:
        print("\n" + "="*60)
        print("STEP 5: Creating summary index (with deduplication)...")
        print("="*60)

        seen = {}
        dedup_summaries = []
        idx_map = []
        for idx, summary in enumerate(summaries):
            if summary not in seen:
                seen[summary] = idx
            if idx == seen[summary]:
                dedup_summaries.append(summary)
                idx_map.append(idx)

        print(f"Deduplicated: {len(summaries)} -> {len(dedup_summaries)} summaries")
        if len(dedup_summaries) < 5:
            print(f"  WARNING: Very few unique summaries ({len(dedup_summaries)}), Step 6 refinement may degrade")

        text_inputs = data.load_and_transform_text(dedup_summaries, "cuda:0")

        with torch.no_grad():
            text_embeddings = self.imagebind_model({ModalityType.TEXT: text_inputs})
            text_vectors = text_embeddings[ModalityType.TEXT].cpu().numpy()

        faiss.normalize_L2(text_vectors)

        index = faiss.IndexFlatIP(INDEX_DIM)
        index.add(text_vectors)

        print(f"Summary index created with {index.ntotal} vectors")
        return index, idx_map

    def step6_refine_scores(self, summaries: List[str], scores: List[float], frame_paths: List[str], summary_index: faiss.Index, idx_map: List[int]) -> Tuple[List[float], Dict]:
        print("\n" + "="*60)
        print(f"STEP 6: Refining scores with {NUM_NEIGHBORS_STEP6} neighbors (two-stage)...")
        print("="*60)

        frames_per_clip = int(CLIP_DURATION * FPS)
        num_clips = len(frame_paths)

        if num_clips < MIN_CLIPS_FOR_REFINEMENT:
            print(f"  WARNING: Only {num_clips} clips (< {MIN_CLIPS_FOR_REFINEMENT}), "
                  f"skipping neighbor refinement (not enough data)")
            return list(scores), {}

        neighbor_clip_indices = {}
        video_captions_nn = {str(i): {} for i in range(num_clips)}
        video_similarity_nn = {str(i): {} for i in range(num_clips)}

        print("Stage 1: Retrieving nearest neighbors...")
        for clip_idx in tqdm(range(0, num_clips), desc="Retrieving NN"):
            clip_center_frame = clip_idx
            start_frame = max(clip_center_frame - frames_per_clip // 2, 0)
            end_frame = min(clip_center_frame + frames_per_clip // 2, len(frame_paths))
            clip_frame_paths = frame_paths[start_frame:end_frame]

            clip_subsample_paths = uniform_temporal_subsample(clip_frame_paths, NUM_SAMPLES)

            vision_inputs = data.load_and_transform_vision_data(clip_subsample_paths, "cuda:0")

            with torch.no_grad():
                vision_embeddings = self.imagebind_model({ModalityType.VISION: vision_inputs})
                vision_vector = vision_embeddings[ModalityType.VISION].cpu().numpy()

            if vision_vector.ndim == 1:
                vision_vector = vision_vector.reshape(1, -1)

            faiss.normalize_L2(vision_vector)
            D, I = summary_index.search(vision_vector, NUM_NEIGHBORS_STEP6)

            for frame_local_idx in range(min(vision_vector.shape[0], NUM_SAMPLES)):
                for nn_idx in range(NUM_NEIGHBORS_STEP6):
                    neighbor_idx = I[frame_local_idx][nn_idx]
                    distance = D[frame_local_idx][nn_idx]

                    if 0 <= neighbor_idx < len(idx_map):
                        orig_clip_idx = idx_map[neighbor_idx]
                        if 0 <= orig_clip_idx < len(summaries):
                            nn_key = f"{frame_local_idx}_{nn_idx}"
                            video_captions_nn[str(clip_idx)][nn_key] = summaries[orig_clip_idx]
                            video_similarity_nn[str(clip_idx)][nn_key] = float(distance)
                            if str(clip_idx) not in neighbor_clip_indices:
                                neighbor_clip_indices[str(clip_idx)] = {}
                            neighbor_clip_indices[str(clip_idx)][nn_key] = orig_clip_idx

        print("Stage 2: Refining scores with exp-weighted average...")
        refined_scores = []
        for clip_idx in tqdm(range(num_clips), desc="Refining scores"):
            frame_scores = []
            frame_similarities = []

            for nn_key in neighbor_clip_indices.get(str(clip_idx), {}):
                nn_clip_idx = neighbor_clip_indices[str(clip_idx)][nn_key]
                frame_scores.append(scores[nn_clip_idx])
                frame_similarities.append(video_similarity_nn[str(clip_idx)][nn_key])

            if frame_scores:
                frame_scores_arr = np.array(frame_scores)
                frame_similarities_arr = np.array(frame_similarities)
                exp_weights = np.exp(frame_similarities_arr)
                frame_weights = exp_weights / np.sum(exp_weights)
                refined_score = float(np.sum(frame_scores_arr * frame_weights))
                refined_score = max(0.0, min(1.0, refined_score))
            else:
                refined_score = max(0.0, min(1.0, scores[clip_idx]))

            refined_scores.append(refined_score)

            if clip_idx < 3:
                print(f"  DEBUG Clip {clip_idx}: refined={refined_score:.4f}, original={scores[clip_idx]:.4f}")

        refined = np.array(refined_scores)
        print(f"Refined {len(refined_scores)} scores")
        print(f"  Refined stats: min={refined.min():.4f}, max={refined.max():.4f}, "
              f"mean={refined.mean():.4f}, std={refined.std():.4f}")
        return refined_scores, video_similarity_nn

    def step7_evaluate(self, scores: List[float], video_fps: float, frame_paths: List[str] = None) -> Tuple[List[Dict], float, float, float, float, float]:
        print("\n" + "="*60)
        print("STEP 7: Final evaluation...")
        print("="*60)

        scores_array = np.array(scores)

        mean_score = float(np.mean(scores_array))
        std_score = float(np.std(scores_array))

        if len(scores_array) < MIN_CLIPS_FOR_STATS:
            threshold = 0.45
            print(f"  WARNING: Only {len(scores_array)} clips (< {MIN_CLIPS_FOR_STATS}), "
                  f"using fixed threshold={threshold}")
        else:
            threshold = max(mean_score, 0.45)

        print(f"Threshold: {threshold:.4f} (mean: {mean_score:.4f}, std: {std_score:.4f})")

        import base64
        anomaly_frames = []
        frame_interval = getattr(self, '_current_interval', FRAME_INTERVAL)
        for idx, score in enumerate(scores):
            if score > threshold:
                timestamp = idx * frame_interval / video_fps if video_fps > 0 else idx * frame_interval / 30
                af = {
                    "frame_idx": idx,
                    "timestamp": round(timestamp, 2),
                    "score": round(float(score), 4)
                }
                if frame_paths and idx < len(frame_paths):
                    af["image_path"] = frame_paths[idx]
                    try:
                        with open(frame_paths[idx], "rb") as fimg:
                            af["image_base64"] = base64.b64encode(fimg.read()).decode("utf-8")
                    except Exception as e:
                        print(f"  WARNING: Failed to read frame image {frame_paths[idx]}: {e}")
                anomaly_frames.append(af)

        anomaly_ratio = len(anomaly_frames) / len(scores) if scores else 0
        max_score = float(max(scores)) if scores else 0
        min_score = float(min(scores)) if scores else 0

        print(f"Detected {len(anomaly_frames)} anomaly frames ({anomaly_ratio*100:.2f}%)")
        print(f"Score range: [{min_score:.4f}, {max_score:.4f}]")

        return anomaly_frames, anomaly_ratio, max_score, threshold, mean_score, std_score

    def analyze_video(self, video_path: str) -> Dict:
        start_time = time.time()
        video_id = str(uuid.uuid4())
        video_name = Path(video_path).name

        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        cap.release()

        if total_frames < 150:
            adaptive_interval = FRAME_INTERVAL_SHORT
        elif total_frames < 300:
            adaptive_interval = FRAME_INTERVAL_MEDIUM
        else:
            adaptive_interval = FRAME_INTERVAL

        print(f"\nVideo: {video_name}, {total_frames} frames, {duration:.1f}s → interval={adaptive_interval}")

        steps_completed = []

        frames, video_fps, sampled_frames = self.step0_extract_frames(video_path, adaptive_interval)
        self._current_interval = adaptive_interval
        steps_completed.append("Step 1: Frame Extraction")

        all_captions = self.step1_generate_captions(frames)
        self._last_captions = all_captions
        steps_completed.append("Step 2: BLIP-2 Captioning (5 models)")

        text_index, file_names = self.step2_create_text_index(all_captions)
        steps_completed.append("Step 3: ImageBind Text Index")

        cleaned_captions = self.step3_clean_captions(all_captions, frames, text_index, file_names)
        self._last_cleaned_captions = cleaned_captions
        steps_completed.append("Step 4: Cross-modal Caption Cleaning")

        summaries, raw_scores = self.step4_llm_summaries_and_scores(cleaned_captions)
        self._last_summaries = summaries
        self._last_raw_scores = raw_scores
        steps_completed.append("Step 5: LLM Summaries & Scoring")

        summary_index, idx_map = self.step5_create_summary_index(summaries)
        steps_completed.append("Step 6: Summary Index Creation")

        refined_scores, similarities = self.step6_refine_scores(summaries, raw_scores, frames, summary_index, idx_map)
        self._last_similarities = similarities
        steps_completed.append("Step 7: Multi-neighbor Score Refinement")

        refined_arr = np.array(refined_scores)
        if refined_arr.std() < 0.005:
            print(f"\n  !!! WARNING: Step 6 collapsed all scores to std={refined_arr.std():.6f}")
            print(f"  !!! Only {len(idx_map)} unique summaries in the index (from {len(summaries)} total)")
            print(f"  !!! Falling back to Step 4 raw scores for anomaly detection")
            final_scores = [s * 1.0 for s in raw_scores]
            self._last_refined_scores = final_scores
        else:
            final_scores = refined_scores
            self._last_refined_scores = refined_scores

        anomaly_frames, anomaly_ratio, max_score, threshold, mean_score, std_score = self.step7_evaluate(final_scores, video_fps, frames)
        steps_completed.append("Step 8: Final Evaluation")

        self._last_anomaly_frames = anomaly_frames
        self._last_frame_paths = frames

        for af in anomaly_frames:
            idx = af["frame_idx"]
            str_idx = str(idx)
            
            caption_found = False
            for model_path in BLIP2_MODELS:
                if model_path in cleaned_captions:
                    model_caps = cleaned_captions[model_path]
                    if isinstance(model_caps, dict) and str_idx in model_caps:
                        nn_data = model_caps[str_idx]
                        if isinstance(nn_data, dict) and "0" in nn_data:
                            caption_text = nn_data["0"].get("caption", "")
                            if caption_text:
                                af["caption"] = caption_text
                                caption_found = True
                                break
                        elif isinstance(nn_data, dict):
                            for key, val in nn_data.items():
                                if isinstance(val, dict) and val.get("caption"):
                                    af["caption"] = val["caption"]
                                    caption_found = True
                                    break
                    elif isinstance(model_caps, list) and idx < len(model_caps):
                        for item in model_caps[idx] if isinstance(model_caps[idx], list) else []:
                            if isinstance(item, dict) and item.get("caption"):
                                af["caption"] = item["caption"]
                                caption_found = True
                                break
            
            if not caption_found and idx < len(all_captions):
                for model_path in BLIP2_MODELS:
                    if model_path in all_captions and idx < len(all_captions[model_path]):
                        cap = all_captions[model_path][idx]
                        if isinstance(cap, list) and len(cap) > 0:
                            af["caption"] = cap[0]
                            break
                        elif isinstance(cap, str):
                            af["caption"] = cap
                            break
            
            af["summary"] = summaries[idx] if idx < len(summaries) else ""
            
            timestamp = af.get("timestamp", 0)
            af["timestamp_str"] = f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}"

        top_anomaly_captions = [f["summary"] for f in sorted(anomaly_frames, key=lambda x: x["score"], reverse=True)[:5]]

        if len(anomaly_frames) > 0:
            summary_text = f"检测到 {len(anomaly_frames)} 个异常帧 (占总视频的 {anomaly_ratio*100:.1f}%)。最高异常评分: {max_score:.2f}"
        else:
            summary_text = "未检测到明显异常。视频显示正常活动。"

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
            "overall_score": round(float(np.mean(final_scores)), 4),
            "max_score": round(max_score, 4),
            "threshold": round(threshold, 4),
            "mean_score": round(mean_score, 4),
            "std_score": round(std_score, 4),
            "summary": summary_text,
            "top_anomaly_captions": top_anomaly_captions,
            "processing_time": round(time.time() - start_time, 2),
            "steps_completed": steps_completed
        }

    def cleanup(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.loaded_blip2_models.clear()
        torch.cuda.empty_cache()

    def export_anomaly_frames(self) -> 'io.BytesIO':
        import zipfile
        import io

        frames = getattr(self, '_last_anomaly_frames', [])
        if not frames:
            raise ValueError("No anomaly frames available. Run analysis first.")

        buf = io.BytesIO()
        descriptions_lines = []
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for af in frames:
                if "image_path" in af and os.path.exists(af["image_path"]):
                    arcname = f"frame_{af['frame_idx']:06d}_score_{af['score']:.4f}.jpg"
                    zf.write(af["image_path"], arcname)
                caption = af.get("caption", "")
                summary = af.get("summary", "")
                descriptions_lines.append(
                    f"帧 #{af['frame_idx']} | 时间: {af.get('timestamp', 0):.1f}s | 评分: {af['score']:.4f}\n"
                    f"  描述: {caption}\n"
                    f"  摘要: {summary}\n"
                )
            desc_content = "\n".join(descriptions_lines)
            zf.writestr("anomaly_frames_descriptions.txt", desc_content)
        buf.seek(0)
        return buf

    def get_intermediate_results(self) -> Dict:
        result = {}

        captions = getattr(self, '_last_captions', {})
        if captions:
            result["captions"] = {}
            for model_path, cap_list in captions.items():
                model_name = os.path.basename(model_path)
                items = []
                for i, cap in enumerate(cap_list):
                    if cap:
                        items.append({"clip_idx": i, "caption": cap})
                result["captions"][model_name] = items

        cleaned = getattr(self, '_last_cleaned_captions', {})
        if cleaned:
            result["cleaned_captions"] = {}
            for model_path, clip_dict in cleaned.items():
                model_name = os.path.basename(model_path)
                items = []
                for clip_idx_str, nn_dict in clip_dict.items():
                    if nn_dict:
                        caps = [{"nn_idx": k, "caption": v.get("caption", ""), "similarity": v.get("similarity", 0)} for k, v in nn_dict.items()]
                        items.append({"clip_idx": int(clip_idx_str), "captions": caps})
                result["cleaned_captions"][model_name] = items

        summaries = getattr(self, '_last_summaries', [])
        raw_scores = getattr(self, '_last_raw_scores', [])
        if summaries:
            result["step4_summaries"] = [{"clip_idx": i, "summary": s, "raw_score": float(raw_scores[i]) if i < len(raw_scores) else None} for i, s in enumerate(summaries)]

        refined_scores = getattr(self, '_last_refined_scores', [])
        if not refined_scores and raw_scores:
            refined_scores = list(raw_scores)
        
        if refined_scores:
            items = []
            for i, score in enumerate(refined_scores):
                item = {"clip_idx": i, "refined_score": float(score)}
                if i < len(raw_scores):
                    item["raw_score"] = float(raw_scores[i])
                    item["score_delta"] = float(score - raw_scores[i])
                else:
                    item["raw_score"] = float(score)
                    item["score_delta"] = 0.0
                items.append(item)
            result["step6_refined_scores"] = items

        return result


class DemoResults:
    DEMO_RESULTS = {
        "ucf_crime": {
            "dataset": "UCF-Crime",
            "roc_auc": 0.3863,
            "pr_auc": 0.0392,
            "description": "LAVAD with Qwen-7B-Chat on UCF-Crime (5 BLIP-2 models)",
            "sample_videos": []
        }
    }

    @classmethod
    def get_results(cls):
        return cls.DEMO_RESULTS["ucf_crime"]

    @classmethod
    def get_sample_videos(cls):
        return cls.DEMO_RESULTS["ucf_crime"]["sample_videos"]