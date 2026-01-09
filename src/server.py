from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
import torch
import numpy as np
import io
from PIL import Image
from .model import AvatarEngine
from .audio import AudioEncoder

app = FastAPI()

# Initialize Engine (Global State)
# We load on startup to avoid latency per request
engine = AvatarEngine()
audio_encoder = AudioEncoder("checkpoints/hubert_quantized.onnx")

@app.post("/generate")
async def generate_avatar(file: UploadFile):
    """
    Input: Audio Blob (WAV/PCM)
    Output: Raw RGB Bytes (or encoded JPEG)
    """
    audio_bytes = await file.read()
    
    # 1. Extract Features (ONNX)
    features = audio_encoder.preprocess(audio_bytes)
    
    # 2. Render (CUDA)
    rendered_tensor = engine.render_frame(features)
    
    # 3. Convert to Image
    # (In prod: Stream H.264 directly via FFmpeg pipe)
    img_np = (rendered_tensor.cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return Response(content=buf.getvalue(), media_type="image/jpeg")