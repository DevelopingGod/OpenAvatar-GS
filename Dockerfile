# Start from NVIDIA PyTorch Base (includes CUDA 11.8)
FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y ffmpeg libsndfile1

# 2. Install Python Dependencies
# We install gsplat first to compile CUDA kernels
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy Code and Checkpoints
COPY . .

# 4. Expose Port
EXPOSE 8000

# 5. Run Server
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]