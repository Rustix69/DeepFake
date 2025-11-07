# Localized Physiological Signal Inconsistency Detection for Deepfake Videos
Minor Project | 2025 | Deepfake Detection using Multi-Region rPPG and Hybrid CNN-Transformer Architecture

---

## 1. Overview

This project focuses on building an advanced deepfake detection system that identifies manipulated videos through physiological inconsistencies rather than traditional pixel or artifact-based methods.

Recent studies (2025) have shown that high-quality deepfakes can preserve global heart rate patterns, making existing rPPG-based methods ineffective. To address this, this project introduces a localized analysis of facial blood flow signals using multi-region remote photoplethysmography (rPPG) and a hybrid deep learning architecture combining CNN and Vision Transformer (ViT).

In essence, this system detects deepfakes by analyzing region-wise inconsistencies in facial pulse patterns that cannot be realistically reproduced by generative models.

---

## 2. Core Concept

Real human faces exhibit synchronized micro-variations in skin tone due to blood flow, measurable as rPPG signals. Deepfakes, even high-quality ones, struggle to maintain consistent localized signals across multiple facial regions.

This system:
- Extracts physiological (rPPG) signals from multiple facial regions.
- Analyzes cross-region consistency and coherence.
- Uses deep learning to classify a video as real or fake.
- Produces interpretable spatial visualizations showing inconsistencies.

---

## 3. System Architecture

### 3.1 Overview
The architecture consists of eight major components forming an end-to-end detection pipeline:

1. Video Ingestion  
2. Face and ROI Extraction  
3. rPPG Signal Extraction  
4. Feature Encoding via CNN  
5. Cross-Region Vision Transformer (ViT)  
6. Spatio-Temporal Fusion  
7. Inconsistency Detection and Visualization  
8. Web API and Interface

---

### 3.2 Detailed Component Description

1. **Video Ingestion**  
   - Supports offline videos, live webcam feeds, and benchmark datasets (DFDC, Celeb-DF v2, FakeAVCeleb, FF++, KoDF).  
   - Frames are extracted at 25–30 FPS with consistent resolution.

2. **Face and ROI Extraction**  
   - Facial landmarks detected using MediaPipe or dlib.  
   - Anatomically significant ROIs defined (forehead, cheeks, jawline, temples).  
   - Optical flow ensures ROI stability across frames.

3. **rPPG Signal Extraction**  
   - Uses ICA, CHROM, or POS algorithms for color-based pulse signal recovery.  
   - Includes motion compensation and illumination normalization.  
   - Applies signal quality gating using SNR thresholds.

4. **Feature Encoding**  
   - Each ROI signal is represented in temporal and spectral forms.  
   - EfficientNet-B4 or MobileNet encodes local physiological patterns.  
   - Extracted features are normalized and fused.

5. **Cross-Region Vision Transformer (ViT)**  
   - Learns dependencies among ROIs to capture localized inconsistencies.  
   - Attention layers identify non-synchronous physiological behavior.

6. **Spatio-Temporal Fusion**  
   - Multi-scale 1D convolutions or transformer blocks integrate temporal features.  
   - Generates a unified feature map representing both spatial and temporal correlations.

7. **Inconsistency Detection and Visualization**  
   - Binary classifier outputs real/fake label and confidence.  
   - Heatmaps generated via Grad-CAM to indicate inconsistent regions.  
   - Visualization aids forensic interpretation.

8. **Web API and Interface**  
   - FastAPI-based inference server with endpoints for batch and live processing.  
   - React or Streamlit interface for video upload, analysis, and result display.  
   - Provides visual outputs like confidence scores and heatmaps.

---

## 4. Data Flow

### Training Phase
1. Load datasets (DFDC, Celeb-DF v2, FakeAVCeleb, KoDF, FF++).  
2. Preprocess frames and extract ROI-based rPPG signals.  
3. Train CNN and ViT modules jointly using classification and consistency losses.  
4. Evaluate cross-dataset generalization and robustness to compression.

### Inference Phase
1. Accept input video or live stream.  
2. Extract ROIs and corresponding rPPG signals.  
3. Encode features and perform cross-region analysis.  
4. Classify and visualize physiological inconsistencies.  
5. Output includes prediction label, confidence, and spatial heatmap.

---

## 5. Technology Stack

Programming Language: Python 3.10+  
Deep Learning: PyTorch or TensorFlow  
Computer Vision: OpenCV, dlib, scikit-image  
Signal Processing: NumPy, SciPy, HeartPy, BioSPPy  
Visualization: Matplotlib, Plotly, Seaborn  
Deployment: FastAPI, ONNX Runtime, TensorRT  
Development Environment: VS Code, Jupyter Notebook, Google Colab Pro

---

## 6. Dataset Details

1. **FakeAVCeleb**  
   Multimodal dataset containing synchronized audio-visual deepfakes.

2. **DFDC (Deepfake Detection Challenge)**  
   Large-scale dataset with over 100,000 manipulated videos of varying quality.

3. **Celeb-DF v2**  
   High-quality celebrity deepfake dataset widely used in research.

4. **KoDF and FF++**  
   Supplementary datasets for cross-domain validation.

5. **Custom Dataset**  
   Optional, with synchronized physiological sensors for rPPG ground truth.

---

## 7. Evaluation Metrics

Detection Metrics: Accuracy, Precision, Recall, F1-score, AUC-ROC  
Localization Metrics: Spatial consistency, region-wise accuracy  
Physiological Metrics: Signal SNR, inter-region correlation  
Performance Metrics: Inference speed, GPU utilization, latency  
Robustness Metrics: Resistance to compression, lighting, and motion artifacts

---

## 8. Risks and Mitigation

- **Dataset Quality**: Use multiple datasets and quality filtering.
- **High Computational Load**: Use Colab Pro or cloud GPU; model pruning and quantization.
- **Signal Noise**: Apply ICA and adaptive filtering.
- **Evaluation Bias**: Conduct cross-dataset validation and balanced sampling.

---

## 9. Expected Outcomes

1. A robust deepfake detection system that can identify even high-quality fakes preserving global heart rate patterns.  
2. Real-time detection capability on consumer hardware.  
3. Explainable forensic visualizations through spatial heatmaps.  
4. Research framework for future physiological-based detection methods.  

---

## 10. References

1. Seibold, C. et al., "High-quality Deepfakes Have a Heart", Frontiers in Imaging, 2025.  
2. Tian, J. & Zhang, L., "STCDePhysio: Spatio-Temporal Consistency of Physiological Signals", SSRN 2025.  
3. Chen, Y. et al., "Deepfake Detection with Spatio-Temporal Consistency and Attention", arXiv 2025.  
4. Li, Y. et al., "Celeb-DF: A Large-scale Dataset for DeepFake Forensics", CVPR 2020.  
5. Rossler, A. et al., "FaceForensics++", ICCV 2019.  

---


## 11. License

This project is for academic and research purposes only.  
All datasets and models used are subject to their respective licenses.

---

## 12. Summary Statement

This project introduces a novel approach to deepfake detection through localized physiological signal analysis.  
By combining computer vision, signal processing, and deep learning, it establishes a robust, explainable, and biologically grounded method for digital media authentication.
