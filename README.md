# Real-to-Sim Vision Pipeline: Feature Extraction & Matching

> **Note:** This repository contains the core architectural implementations and deep learning pipelines developed during my undergraduate research at ROMER (METU).

##  Research Overview
Bridging the gap between real-world visual data and simulated environments (Real-to-Sim) is a critical challenge in modern robotics and AI. This project implements an end-to-end computer vision pipeline that robustly segments, extracts, and matches features across domains with high variance in texture and lighting.

##  Model Architecture & Pipeline
The pipeline integrates three state-of-the-art deep learning architectures, distributing the workload from pixel-level segmentation to global image representation:

### 1. Semantic Segmentation (U-Net)
* **Role:** Pre-processing and isolating regions of interest.
* **Implementation:** Utilized **U-Net** to segment critical elements within both real and simulated frames, reducing background noise and focusing the subsequent feature extractors on relevant data.

### 2. Local Feature Matching (LoFTR)
* **Role:** Finding precise point-to-point correspondences.
* **Implementation:** Deployed **LoFTR** (Local Feature Matching with Transformers). Unlike traditional CNN-based detectors, LoFTR's self-attention mechanism allowed us to extract dense matches even in low-texture environments common in simulated domains.

### 3. Global Descriptor Aggregation (NetVLAD)
* **Role:** Image retrieval and global context matching.
* **Implementation:** Integrated a **NetVLAD** layer to pool the extracted local features into a compact, highly discriminative global descriptor, enabling accurate place recognition and domain alignment.

##  Technology Stack
* **Deep Learning Frameworks:** PyTorch, TensorFlow
* **Models:** U-Net, LoFTR, NetVLAD
* **Data Manipulation:** Python (NumPy, Pandas, OpenCV)
* **Domain:** Real-to-Sim, Computer Vision, Academic Research

# Core Deep Learning & Vision
torch>=1.10.0
torchvision>=0.11.0
numpy
opencv-python

# Feature Extraction & Matching Dependencies
kornia>=0.6.11
scipy
h5py

# Utilities
tqdm>=4.36.0
