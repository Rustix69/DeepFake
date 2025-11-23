# Setup Guide

Complete installation guide for the Deepfake Detection System with GPU support.

---

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (RTX 2060 or better recommended)
- **VRAM**: Minimum 4GB (RTX 3050 or better)
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ free space (for dataset + outputs)

### Software
- **OS**: Ubuntu 20.04+ / Linux
- **Python**: 3.8 - 3.12
- **CUDA**: 12.1+ (installed via NVIDIA drivers)
- **Git**: For cloning the repository

---

## Installation Steps

### 1. NVIDIA GPU Setup (CRITICAL for Training)

#### Check if GPU is detected:
```bash
nvidia-smi
```

If you see an error, follow these steps:

#### Install NVIDIA Drivers:
```bash
# Check available drivers
ubuntu-drivers devices

# Install recommended driver (usually nvidia-driver-535)
sudo apt install nvidia-driver-535 -y

# Reboot
sudo reboot
```

#### After Reboot - Verify:
```bash
nvidia-smi
```

**Expected output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name                     | Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce RTX 3050  | 00000000:01:00.0 Off |                  N/A |
+-----------------------------------------------------------------------------+
```

**Troubleshooting:**
- If you see MOK enrollment screen during boot, select "Enroll MOK" and follow the prompts
- Or disable Secure Boot in BIOS (easier option)

---

### 2. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/DeepFake.git
cd DeepFake
```

---

### 3. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

**Note**: Always activate the virtual environment before running commands!

---

### 4. Install PyTorch with CUDA Support

**IMPORTANT**: Install PyTorch with CUDA BEFORE other dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify GPU is detected by PyTorch:**
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected output:**
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
```

---

### 5. Install Other Dependencies

```bash
pip install opencv-python mediapipe numpy scipy pandas matplotlib seaborn scikit-learn tqdm timm einops tensorboard pyyaml pillow
```

**Optional but recommended:**
```bash
pip install jupyter ipykernel wandb
```

---

### 6. Verify Installation

```bash
python -c "
import torch
import cv2
import mediapipe
import numpy
import scipy
print('✅ All core packages installed successfully!')
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
"
```

---

## Download Dataset

### Option 1: Manual Download (Recommended)

1. Visit: https://github.com/yuezunli/celeb-deepfakeforensics
2. Download `Celeb-DF-v2` (59.3 GB)
3. Extract to project root:
   ```bash
   unzip Celeb-DF-v2.zip -d /home/traderx/DeepFake/
   ```

### Option 2: Automated Download (If available)

```bash
python src/data/download_dataset.py --output_dir Celeb-DF-v2
```

### Verify Dataset Structure:

```bash
ls -la Celeb-DF-v2/
```

**Expected:**
```
Celeb-DF-v2/
├── Celeb-real/         # 590 real celebrity videos
├── Celeb-synthesis/    # 5,639 fake videos
├── YouTube-real/       # 300 real YouTube videos
└── List_of_testing_videos.txt
```

---

## Quick Test

Run a quick test to ensure everything is working:

```bash
python -c "
import sys
sys.path.append('src')
from preprocessing.face_detection import FaceDetector
import cv2

print('✅ Importing modules... OK')
detector = FaceDetector()
print('✅ Face detector initialized... OK')
print('✅ Setup complete! Ready to preprocess and train.')
"
```

---

## Common Issues

### Issue 1: `CUDA Available: False`

**Solution**: Reinstall PyTorch with CUDA:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### Issue 2: `nvidia-smi` not working

**Solution**: Reinstall NVIDIA drivers:
```bash
sudo apt remove --purge '^nvidia-.*' -y
sudo apt install nvidia-driver-535 -y
sudo reboot
```

---

### Issue 3: Out of Memory during training

**Solution**: Reduce batch size:
```bash
python train.py --batch_size 8  # instead of 16
```

---

### Issue 4: `ModuleNotFoundError`

**Solution**: Ensure virtual environment is activated:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

## Next Steps

Once setup is complete, proceed to:
1. **Preprocess Dataset**: See `QUICK_START_GUIDE.md`
2. **Train Model**: Run `python train.py`
3. **Evaluate Results**: Check `outputs/` directory

---

## GPU Performance Tips

1. **Monitor GPU usage during training:**
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Optimal batch size for RTX 3050 (4GB VRAM):**
   - Batch size: 16 (recommended)
   - If OOM error: reduce to 8

3. **Expected training time:**
   - Preprocessing: 2-3 hours
   - Training (50 epochs): 30-60 minutes
   - Total: ~3-4 hours

---

## Support

If you encounter issues not listed here:
1. Check GPU is working: `nvidia-smi`
2. Check CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
3. Verify dataset structure: `ls -la Celeb-DF-v2/`
4. Check logs: `cat logs/training.log`

---

**Last Updated**: November 2025
**Tested On**: Ubuntu 24.04, RTX 3050, CUDA 12.2, PyTorch 2.5.1

