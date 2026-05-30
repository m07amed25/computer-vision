<div align="center">

# 🧠 Computer Vision Portfolio

### Advanced Projects in Deep Learning, OpenCV & Image Processing

*A curated collection of computer vision projects spanning classical image processing,
modern deep learning architectures, and real-world deployment.*

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## 📌 Overview

This repository documents my research and engineering work in **computer vision**, developed
during my **Master's degree** studies. It brings together **classical OpenCV techniques** and
**state-of-the-art deep learning models** to solve problems in detection, segmentation,
recognition, tracking, and generative imaging.

Each project includes a clear problem statement, the methods used, results, and reproducible code.

---

## 🗂️ Table of Contents

- [Tech Stack](#-tech-stack)
- [Project Categories](#-project-categories)
  - [1. Image Classification](#1-image-classification)
  - [2. Object Detection](#2-object-detection)
  - [3. Semantic & Instance Segmentation](#3-semantic--instance-segmentation)
  - [4. Face Analysis & Recognition](#4-face-analysis--recognition)
  - [5. Pose Estimation & Action Recognition](#5-pose-estimation--action-recognition)
  - [6. Object Tracking](#6-object-tracking)
  - [7. Generative Models (GANs & Diffusion)](#7-generative-models-gans--diffusion)
  - [8. Classical OpenCV & Image Processing](#8-classical-opencv--image-processing)
  - [9. 3D Vision & Depth Estimation](#9-3d-vision--depth-estimation)
  - [10. OCR & Document Understanding](#10-ocr--document-understanding)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Results & Benchmarks](#-results--benchmarks)
- [Datasets](#-datasets)
- [Roadmap](#-roadmap)
- [References](#-references)
- [Contact](#-contact)

---

## 🛠️ Tech Stack

| Category | Tools & Libraries |
|----------|-------------------|
| **Languages** | Python, C++ |
| **Deep Learning** | PyTorch, TensorFlow, Keras |
| **Computer Vision** | OpenCV, scikit-image, Pillow, Albumentations |
| **Models / Zoos** | YOLOv8, Detectron2, MMDetection, Hugging Face Transformers |
| **Data Science** | NumPy, Pandas, Matplotlib, Seaborn, scikit-learn |
| **Deployment** | ONNX, TensorRT, Flask, FastAPI, Streamlit, Docker |
| **Experiment Tracking** | Weights & Biases, TensorBoard, MLflow |

---

## 📂 Project Categories

### 1. Image Classification
> Building and fine-tuning CNNs and Vision Transformers for image recognition.

- **CNN from Scratch** — Implemented LeNet, VGG, and ResNet architectures using PyTorch.
- **Transfer Learning** — Fine-tuned EfficientNet & ResNet-50 on custom datasets.
- **Vision Transformers (ViT)** — Compared ViT vs. CNN performance on CIFAR-100.

`Tech:` PyTorch · timm · Albumentations

---

### 2. Object Detection
> Locating and classifying multiple objects in images and video streams.

- **Real-Time Detection with YOLOv8** — Custom-trained detector for traffic/road objects.
- **Faster R-CNN** — Two-stage detector implemented with Detectron2.
- **SSD MobileNet** — Lightweight detection optimized for edge devices.

`Tech:` YOLOv8 · Detectron2 · OpenCV DNN

---

### 3. Semantic & Instance Segmentation
> Pixel-level scene understanding.

- **U-Net for Medical Imaging** — Tumor/cell segmentation on biomedical scans.
- **Mask R-CNN** — Instance segmentation on the COCO dataset.
- **DeepLabV3+** — Semantic segmentation for urban street scenes (Cityscapes).

`Tech:` PyTorch · segmentation-models · Detectron2

---

### 4. Face Analysis & Recognition
> Detecting, verifying, and analyzing human faces.

- **Face Detection & Landmarks** — Dlib & MediaPipe facial landmark extraction.
- **Face Recognition** — Embedding-based verification with FaceNet / ArcFace.
- **Emotion Recognition** — CNN classifier for facial expression analysis (FER-2013).

`Tech:` Dlib · MediaPipe · OpenCV · FaceNet

---

### 5. Pose Estimation & Action Recognition
> Understanding human body movement.

- **Human Pose Estimation** — Keypoint detection with MediaPipe & OpenPose.
- **Gesture Recognition** — Hand-tracking control system.
- **Action Recognition** — Temporal models (LSTM / 3D-CNN) on video sequences.

`Tech:` MediaPipe · OpenPose · PyTorch

---

### 6. Object Tracking
> Following objects across video frames.

- **Multi-Object Tracking** — DeepSORT + YOLO for pedestrian/vehicle tracking.
- **Optical Flow** — Lucas–Kanade & Farnebäck motion estimation with OpenCV.
- **Centroid & Kalman Tracking** — Classical tracking baselines.

`Tech:` DeepSORT · OpenCV · Kalman Filter

---

### 7. Generative Models (GANs & Diffusion)
> Synthesizing and transforming images.

- **DCGAN & StyleGAN** — Generating realistic synthetic images.
- **CycleGAN** — Unpaired image-to-image translation (e.g., style transfer).
- **Stable Diffusion** — Text-to-image generation and inpainting experiments.

`Tech:` PyTorch · Hugging Face Diffusers

---

### 8. Classical OpenCV & Image Processing
> Foundations of computer vision without deep learning.

- **Image Filtering & Edge Detection** — Sobel, Canny, Gaussian, morphological ops.
- **Feature Matching** — SIFT, SURF, ORB keypoint detection & homography.
- **Panorama Stitching** — Multi-image stitching using feature matching.
- **Document Scanner** — Perspective transform & adaptive thresholding.

`Tech:` OpenCV · NumPy · scikit-image

---

### 9. 3D Vision & Depth Estimation
> Reconstructing depth and geometry from images.

- **Stereo Depth Estimation** — Disparity maps from stereo image pairs.
- **Monocular Depth (MiDaS)** — Single-image depth prediction.
- **Camera Calibration** — Intrinsic/extrinsic parameter estimation.

`Tech:` OpenCV · MiDaS · Open3D

---

### 10. OCR & Document Understanding
> Extracting text and structure from images.

- **Text Detection & Recognition** — EAST + Tesseract / EasyOCR pipeline.
- **Handwritten Digit Recognition** — CNN on MNIST.
- **License Plate Recognition** — Detection + OCR end-to-end system.

`Tech:` Tesseract · EasyOCR · OpenCV

---

## 📁 Repository Structure

```
computer-vision/
├── 01-image-classification/
├── 02-object-detection/
├── 03-segmentation/
├── 04-face-analysis/
├── 05-pose-estimation/
├── 06-object-tracking/
├── 07-generative-models/
├── 08-classical-opencv/
├── 09-3d-vision/
├── 10-ocr/
├── datasets/            # data download scripts & links (not raw data)
├── utils/               # shared helpers (data loaders, metrics, viz)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- (Optional) NVIDIA GPU with CUDA for training

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/computer-vision.git
cd computer-vision

# Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running a Project

```bash
cd 02-object-detection
python detect.py --source path/to/image_or_video --weights weights/best.pt
```

> Each project folder contains its own `README.md` with detailed instructions.

---

## 📊 Results & Benchmarks

| Project | Model | Dataset | Metric | Score |
|---------|-------|---------|--------|-------|
| Image Classification | ResNet-50 | CIFAR-100 | Accuracy | 78.4% |
| Object Detection | YOLOv8 | Custom | mAP@0.5 | 0.91 |
| Segmentation | U-Net | Medical | Dice | 0.89 |
| Face Recognition | ArcFace | LFW | Accuracy | 99.2% |
| Depth Estimation | MiDaS | KITTI | RMSE | 4.1 |

> *Numbers are illustrative placeholders — replace with your actual experiment results.*

---

## 🗃️ Datasets

Public datasets used across projects:

- [ImageNet](https://www.image-net.org/) · [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [COCO](https://cocodataset.org/) · [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
- [Cityscapes](https://www.cityscapes-dataset.com/) · [KITTI](http://www.cvlibs.net/datasets/kitti/)
- [LFW](http://vis-www.cs.umass.edu/lfw/) · [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) · [MNIST](http://yann.lecun.com/exdb/mnist/)

> ⚠️ Raw datasets are **not** stored in this repo. See `datasets/` for download scripts.

---

## 🧭 Roadmap

- [ ] Add self-supervised learning experiments (SimCLR, DINO)
- [ ] Deploy models as REST APIs with FastAPI + Docker
- [ ] Add ONNX / TensorRT optimization benchmarks
- [ ] Integrate Vision-Language models (CLIP, BLIP)
- [ ] Publish interactive demos on Hugging Face Spaces

---

## 📚 References

- Goodfellow et al., *Deep Learning*, MIT Press, 2016
- Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed.
- Redmon et al., *YOLO: You Only Look Once*
- He et al., *Deep Residual Learning for Image Recognition (ResNet)*
- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*

---

## 📬 Contact

**[Your Name]** — M.Sc. Computer Vision / Machine Learning

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/your-username)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:your.email@example.com)

---

<div align="center">

⭐ *If you find this portfolio useful, consider giving it a star!*

</div>
# computer-vision
