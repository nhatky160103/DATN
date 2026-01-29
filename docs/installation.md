# Installation Guide

[← Back to Main README](../README.md)

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Steps](#installation-steps)
- [Dataset Setup](#dataset-setup)
- [Model Weights](#model-weights)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware Requirements

**Minimum:**
- CPU: Quad-core processor (Intel i5 or equivalent)
- RAM: 8GB
- Storage: 10GB free space
- Webcam: 720p resolution

**Recommended:**
- CPU: 8-core processor (Intel i7/i9 or AMD Ryzen 7/9)
- GPU: NVIDIA GPU with 4GB+ VRAM (GTX 1060 or better)
- RAM: 16GB+
- Storage: 20GB+ SSD
- Webcam: 1080p resolution

### Software Requirements

- **Operating System:** Linux (Ubuntu 18.04+), Windows 10+, or macOS 10.14+
- **Python:** 3.8 or newer
- **CUDA:** 10.2+ (for GPU support)
- **cuDNN:** 7.6+ (for GPU support)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/DATN.git
cd DATN
```

### 2. Create Virtual Environment

**Using venv:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n face-recognition python=3.8
conda activate face-recognition
```

### 3. Install Dependencies

**For CPU-only installation:**
```bash
pip install -r requirements.txt
```

**For GPU installation:**
```bash
# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install other dependencies
pip install -r requirements.txt
```

### 4. Install Additional System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
```

**macOS:**
```bash
brew install opencv
```

**Windows:**
- No additional system dependencies required for most setups

## Dataset Setup

### Download Datasets

All datasets and pre-trained models are available at:
📦 [OneDrive Link](https://husteduvn-my.sharepoint.com/:f:/g/personal/ky_dn215410_sis_hust_edu_vn/Etlu7CZEWr5Ao1owHA9pOk0B-wwess_BZfVLEbZTcaWSvw?e=gVMQTf)

### Dataset Structure

```
data/
├── CASIA-WebFace/          # Training dataset (optional)
│   ├── 0000001/
│   ├── 0000002/
│   └── ...
├── MS1MV3/                 # Large-scale training (optional)
│   ├── images/
│   └── labels/
└── validation/             # Validation datasets
    ├── lfw/
    ├── cfp_fp/
    ├── agedb_30/
    └── ...
```

### Download and Extract

```bash
# Create data directory
mkdir -p data

# Download from OneDrive and extract
# (Manual download recommended due to size)

# For validation datasets only:
cd data
# Extract validation.tar.gz here
tar -xzf validation.tar.gz
```

## Model Weights

### Download Pre-trained Models

Download pre-trained model weights from the OneDrive link and place them in the `models/` directory.

### Recommended Models for Production

**For Real-time CPU Deployment:**
```bash
models/
└── Recognition/
    └── Arcface_torch/
        └── weights/
            └── r50_lite_ms1mv3_cdml.pth  # Recommended
```

**For High Accuracy (GPU):**
```bash
models/
└── Recognition/
    └── Arcface_torch/
        └── weights/
            └── r100_ms1mv3_cdml.pth
```

### Required Supporting Models

Download these essential models for the pipeline:

```bash
models/
├── Detection/
│   ├── mtcnn/
│   │   ├── pnet.npy
│   │   ├── rnet.npy
│   │   └── onet.npy
│   └── blazeface/
│       └── blazeface.pth
├── Anti_spoof/
│   ├── fasnet.pth
│   └── minifasnet.pth
└── LightQNet/
    └── lightqnet.pth
```

### Model Download Script

```bash
# Run the model download script
python scripts/download_models.py --model all

# Or download specific models
python scripts/download_models.py --model r50_lite
python scripts/download_models.py --model r100
```

## Configuration

### Firebase Setup

1. **Create Firebase Project:**
   - Go to [Firebase Console](https://console.firebase.google.com/)
   - Create a new project
   - Enable Realtime Database

2. **Get Configuration:**
   - Project Settings → General → Your apps
   - Download service account key (JSON)

3. **Update Config:**

```yaml
# config.yaml
firebase:
  credentials_path: "path/to/serviceAccountKey.json"
  database_url: "https://your-project.firebaseio.com"
  storage_bucket: "your-project.appspot.com"
```

### Cloudinary Setup

1. **Create Cloudinary Account:**
   - Sign up at [Cloudinary](https://cloudinary.com/)
   - Get your cloud name, API key, and API secret

2. **Update Config:**

```yaml
# config.yaml
cloudinary:
  cloud_name: "your-cloud-name"
  api_key: "your-api-key"
  api_secret: "your-api-secret"
```

### System Configuration

Edit `config.yaml` to customize the system:

```yaml
# Pipeline Configuration
pipeline:
  detection_model: "blazeface"  # or "mtcnn"
  quality_threshold: 0.5
  anti_spoofing: true
  embedding_model: "r50_lite"   # r18_lite, r50_lite, r100_lite, r100
  device: "cuda"                # or "cpu"

# Recognition Settings
recognition:
  similarity_metric: "cosine"   # or "euclidean"
  threshold: 0.705
  min_frames: 3
  face_size: [112, 112]

# Camera Settings
camera:
  device_id: 0
  resolution: [1280, 720]
  fps: 30

# Database
firebase:
  credentials_path: "credentials/serviceAccountKey.json"
  database_url: "https://your-project.firebaseio.com"

# Storage
cloudinary:
  cloud_name: "your-cloud-name"
  api_key: "your-api-key"
  api_secret: "your-api-secret"

# Paths
paths:
  model_dir: "models"
  data_dir: "data"
  output_dir: "output"
  log_dir: "logs"
```

## Verification

### Test Installation

```bash
# Run system check
python scripts/check_installation.py
```

Expected output:
```
✓ Python version: 3.8.10
✓ PyTorch installed: 2.0.1
✓ CUDA available: True (11.8)
✓ Required models found
✓ Firebase connection: OK
✓ Cloudinary connection: OK
✓ Camera accessible
```

### Run Quick Test

```bash
# Test face detection
python scripts/test_detection.py --image test_images/sample.jpg

# Test recognition pipeline
python scripts/test_recognition.py --image test_images/sample.jpg

# Test full system
python scripts/test_system.py
```

### Verify Model Loading

```bash
# Test model inference
python -c "
from models.Recognition.Arcface_torch.backbones import get_model
import torch

model = get_model('r50_lite', fp16=False)
model.load_state_dict(torch.load('models/Recognition/Arcface_torch/weights/r50_lite_ms1mv3_cdml.pth'))
print('Model loaded successfully!')
"
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```yaml
# Reduce batch size in config.yaml
pipeline:
  batch_size: 1  # Reduce if needed
  
# Or switch to CPU
pipeline:
  device: "cpu"
```

#### 2. OpenCV Camera Issues

**Error:** `Cannot open camera`

**Solution:**
```bash
# Linux: Check camera permissions
sudo usermod -a -G video $USER
# Logout and login again

# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Failed')"
```

#### 3. Firebase Connection Failed

**Error:** `Failed to connect to Firebase`

**Solution:**
- Verify `serviceAccountKey.json` path is correct
- Check database URL format
- Ensure internet connection
- Verify Firebase project settings

#### 4. Module Import Errors

**Error:** `ModuleNotFoundError: No module named 'xxx'`

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Or install specific package
pip install xxx
```

#### 5. Model File Not Found

**Error:** `FileNotFoundError: Model weights not found`

**Solution:**
```bash
# Check model paths
ls -la models/Recognition/Arcface_torch/weights/

# Re-download models
python scripts/download_models.py --model all
```

#### 6. Slow Inference on CPU

**Issue:** Processing takes too long

**Solution:**
- Use lighter model: Set `embedding_model: "r18_lite"` in config
- Reduce image resolution
- Enable batch processing for MTCNN
- Consider GPU acceleration

### Performance Optimization

**For CPU:**
```yaml
pipeline:
  embedding_model: "r18_lite"  # Fastest
  detection_model: "blazeface"  # Faster than MTCNN
  batch_size: 1
```

**For GPU:**
```yaml
pipeline:
  embedding_model: "r100_lite"  # or "r100" for best accuracy
  detection_model: "mtcnn"
  batch_size: 4
  device: "cuda"
```

### Getting Help

If you encounter issues not covered here:

1. Check existing GitHub issues
2. Review system logs in `logs/` directory
3. Run diagnostic script: `python scripts/diagnose.py`
4. Create a new issue with:
   - Error message
   - System information
   - Configuration file
   - Steps to reproduce

---

[💻 View Usage Guide →](usage.md)

[← Results](results.md) | [Usage →](usage.md)

[← Back to Main README](../README.md)
