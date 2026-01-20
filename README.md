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

Clone the repository:

git clone https://github.com/YOUR_USERNAME/image-classification-resnet50-enhancement.git
cd image-classification-resnet50-enhancement
Install Python dependencies:

pip install -r requirements.txt

# Usage

## 1. Training the Model

Run the training script located in the `python/` folder:

python python/train.py
This will:

Load images from the dataset

Split them into train, validation, and test sets

Train the ResNet50 + Inception-style head model in two stages:

Frozen backbone

Fine-tuning

Save the best models in results/models/

Plot training/validation accuracy and loss curves

## 2. Patch-Based Inference
Use the script python/inference.py to perform patch-based prediction on new images:

python python/inference.py
Patch-based inference allows detection of:

Major class: the dominant class in the image

Secondary classes: classes covering significant areas

Visualization of patch predictions as a grid is also supported.

## 3. Evaluate on Test Set
Evaluation is automatically done after training, including metrics such as:

Accuracy

Macro F1 score

Top-2 accuracy

AUC (One-vs-Rest)

Confusion matrices

## Requirements

The required Python packages are listed in `requirements.txt`:

- `tensorflow==2.13.0`
- `keras==2.13.1`
- `numpy==1.25.0`
- `pandas==2.1.0`
- `matplotlib==3.8.0`
- `seaborn==0.12.3`
- `scikit-learn==1.3.0`

You can install all dependencies using:

pip install -r requirements.txt

---

## Paper

The original research paper is included as `paper.pdf`. It explains in detail the:

- Architecture
- Dataset
- Inference approach




