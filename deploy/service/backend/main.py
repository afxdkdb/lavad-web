import os
import sys
import torch
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "libs" / "ImageBind"))

from deploy.service.backend.models import (
    VideoUploadResponse,
    DetectionResult,
    HealthResponse,
    ModelStatus,
    VideoInfo
)
from deploy.service.backend.lavad_pipeline import LAVADPipeline, DemoResults

pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    try:
        pipeline = LAVADPipeline()
        pipeline.load_models()
        print("All models loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load models at startup: {e}")
        print("Will use demo mode. Models will be loaded on first request.")
    yield
    if pipeline:
        pipeline.cleanup()


app = FastAPI(
    title="LAVAD Video Anomaly Detection API",
    description="Video Anomaly Detection API based on LLM and VLM models (CVPR 2024)",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0

    model_status_dict = {"blip2_loaded": False, "qwen_loaded": False, "imagebind_loaded": False}
    if pipeline:
        model_status_dict = pipeline.get_model_status()

    return HealthResponse(
        status="healthy" if gpu_available else "degraded",
        gpu_available=gpu_available,
        gpu_count=gpu_count,
        model_loaded=all(model_status_dict.values())
    )


@app.get("/model_status", response_model=ModelStatus)
async def get_model_status():
    if not pipeline:
        return ModelStatus(
            blip2_loaded=[],
            blip2_available=[],
            qwen_loaded=False,
            imagebind_loaded=False
        )
    status = pipeline.get_model_status()
    return ModelStatus(
        blip2_loaded=status.get('blip2_loaded', []),
        blip2_available=status.get('available_blip2_models', []),
        qwen_loaded=status.get('qwen_loaded', False),
        imagebind_loaded=status.get('imagebind_loaded', False)
    )


@app.get("/demo_results")
async def get_demo_results():
    return {
        "dataset": "UCF-Crime",
        "roc_auc": 0.7471008133366012,
        "pr_auc": 0.26,
        "description": "LAVAD with Qwen-7B-Chat model trained on UCF-Crime dataset"
    }


@app.get("/sample_videos")
async def get_sample_videos():
    return DemoResults.get_sample_videos()


@app.post("/upload", response_model=VideoUploadResponse)
async def upload_video(video: UploadFile = File(...)):
    if not video.filename.endswith(('.mp4', '.mkv', '.mov')):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    temp_dir = Path(tempfile.mkdtemp())
    temp_video_path = temp_dir / video.filename

    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        video_id = str(hash(video.filename))

        return VideoUploadResponse(
            video_id=video_id,
            message=f"Video {video.filename} uploaded successfully",
            status="uploaded"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        pass


@app.post("/analyze", response_model=DetectionResult)
async def analyze_video(video: UploadFile = File(...)):
    if not video.filename.endswith(('.mp4', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    temp_dir = Path(tempfile.mkdtemp())
    temp_video_path = temp_dir / video.filename

    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        global pipeline
        if pipeline is None:
            pipeline = LAVADPipeline()
            try:
                pipeline.load_models()
            except Exception as e:
                print(f"Could not load models: {e}")
                raise HTTPException(
                    status_code=503,
                    detail="Models not available. Please ensure BLIP-2 and Qwen models are installed."
                )

        result = pipeline.analyze_video(str(temp_video_path))

        return DetectionResult(**result)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/intermediate_results")
async def get_intermediate_results():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        return pipeline.get_intermediate_results()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/video_info/{video_id}")
async def get_video_info(video_id: str):
    return {"video_id": video_id, "status": "placeholder"}


@app.post("/export_anomaly_frames")
async def export_anomaly_frames():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized. Run analysis first.")

    frames = getattr(pipeline, '_last_anomaly_frames', [])
    if not frames:
        raise HTTPException(status_code=404, detail="No anomaly frames available. Run analysis first.")

    try:
        result = pipeline.export_anomaly_frames()
        return StreamingResponse(
            result,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=anomaly_frames.zip"}
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)