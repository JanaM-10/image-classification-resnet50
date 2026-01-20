# Image Classification using ResNet-50 with Architectural Enhancement

## Overview
This deep learning project performs multi-class image classification in complex scenes using a pre-trained ResNet-50 backbone enhanced with a lightweight Inception-style classification head. It implements **patch-based inference** to detect both primary and secondary classes within an image.

The repository includes:
- Training scripts
- Patch-based inference code
- Dataset organization guidance
- Pre-trained models saving and evaluation

---

## Repository Structure

image-classification-resnet50-enhancement/
│
├─ README.md
├─ requirements.txt
├─ paper.pdf
|   └─ Image_Classification_ResNet50.pdf
├─ python/
│   ├─ train.py
│   ├─ inference.py
│   ├─ utils.py
│   └─ models/
│       └─ resnet50_inception.py
└─ dataset/
    └─ README.md


---

## Dataset

The dataset (~2GB) is **not included** due to GitHub size limits.  

You can download it from **Google Drive**:  

[Download Dataset](#)  

After downloading, extract the zip and make sure the folder structure is:

dataset/
├─ Buildings/
├─ Cars/
├─ Labs/
├─ People/
└─ Trees/
---

## Installation
ت
Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/image-classification-resnet50-enhancement.git
cd image-classification-resnet50-enhancement

Install Python dependencies:
pip install -r requirements.txt

## Usage

### 1. Training the Model

Run the training script located in `python/`:

```bash
python python/train.py

ى



