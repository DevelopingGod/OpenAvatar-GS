# OpenAvatar-GS: Real-Time Audio-Driven Avatars via 3D Gaussian Splatting

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10-green)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch_|_GSplat-red)](https://pytorch.org/)

**OpenAvatar-GS** is a high-fidelity, real-time talking head synthesis system. By leveraging **3D Gaussian Splatting (3DGS)** instead of traditional NeRFs, this pipeline achieves **30+ FPS** rendering on consumer-grade GPUs (T4/RTX 3060) with photorealistic quality.

This repository serves as the official reference implementation for the **Task 3: Technical Assignment**.

## 🚀 Key Features

* **Real-Time Inference:** Renders at >30 FPS via CUDA-accelerated rasterization (`gsplat`).
* **Explicit 3D Geometry:** Handles head rotation and depth naturally, avoiding the "flat" look of 2D GANs.
* **Low-Latency Audio:** Uses an optimized INT8 Quantized HuBERT encoder via ONNX Runtime.
* **Production Ready:** Dockerized environment for easy deployment.

## 🛠️ Architecture

The system follows a modular "Audio-to-Motion-to-Splat" architecture:
1.  **Audio Encoder:** Extracts prosodic features from raw audio.
2.  **Motion Policy:** A lightweight MLP/Transformer predicts deformation deltas for the Gaussian cloud.
3.  **Rasterizer:** The deformed Gaussians are sorted and splatted to the screen.

*(See `docs/architecture.png` for the full diagram)*

## 📦 Installation

### Prerequisites
* Linux (Ubuntu 20.04+)
* NVIDIA GPU (Min 6GB VRAM)
* CUDA 11.8+

### Setup via Conda
```bash
git clone [https://github.com/YOUR_USERNAME/OpenAvatar-GS.git](https://github.com/YOUR_USERNAME/OpenAvatar-GS.git)
cd OpenAvatar-GS

conda create -n avatar_gs python=3.10
conda activate avatar_gs
```
### Install PyTorch with CUDA support
```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
```

# Install dependencies
```bash
pip install -r requirements.txt
```

### Setup via Docker (Recommended)
``` Bash

docker build -t openavatar-gs .
docker run --gpus all -p 8000:8000 openavatar-gs
```

### 🖥️ Usage
1. Start the Inference Server
```Bash
uvicorn src.server:app --host 0.0.0.0 --port 8000
```
2. Send an Audio Clip
You can test the API using curl:
```Bash

curl -X POST "http://localhost:8000/generate" \
  -H "accept: image/jpeg" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_audio.wav" \
  --output result_frame.jpg
```

# 📂 Project Structure
├── checkpoints/       # Pre-trained models (Download via script)
├── src/
│   ├── model.py       # Core AvatarEngine and MotionPolicy
│   ├── audio.py       # ONNX Audio Processor
│   └── server.py      # FastAPI Interface
├── Dockerfile         # Deployment container
└── requirements.txt   # Python deps

# 📜 License
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

🙏 Acknowledgements
gsplat for the rasterization backend.

HuBERT for audio feature extraction.
