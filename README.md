# Fire and Smoke Detection Using YOLOv8

## Project Overview

This project investigates the use of YOLOv8 object detection models for automatic fire and smoke detection in images.

The main objective is to analyze how different model architectures and training strategies affect detection performance. In addition to model training, the project includes dataset analysis, experiment design, performance comparison, and engineering evaluation.

The project was developed as part of the ADA447 Deep Learning course.

---

## Problem Statement

Early fire detection plays a critical role in reducing environmental, economic, and human losses caused by wildfires and industrial fires.

Traditional monitoring systems often rely on human supervision, which may delay detection. This project explores how Deep Learning and Computer Vision techniques can be used to automatically detect fire and smoke in images using YOLOv8 object detection models.

---

## Dataset

This project uses the Fire and Smoke Detection Dataset from Kaggle.

Dataset source:

https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo/data

Classes:

* Smoke
* Fire

The dataset is organized in YOLO format and split into:

* Train
* Validation
* Test

> Note: The dataset is not included in this repository due to its size. Please download it from the source link above.

---

## Exploratory Data Analysis (EDA)

Before training, several analyses were performed to better understand the dataset:

* Dataset audit and integrity checks
* Class distribution analysis
* Bounding box size analysis
* Sample visualization with annotations

These analyses helped identify class balance, object size distribution, and potential dataset challenges before model training.

---

## Experiments

Several experiments were conducted to evaluate the effect of different architectures and training configurations.

### Experiment 1: YOLOv8n Baseline

A baseline model was trained using the pretrained YOLOv8n architecture. This experiment served as the reference point for all subsequent comparisons.

### Experiment 2: YOLOv8s Baseline

The YOLOv8s architecture was trained and compared against YOLOv8n to evaluate the trade-off between model complexity and detection performance.

### Experiment 3: Augmentation Disabled

All augmentation techniques were disabled during training. This experiment was designed to measure the contribution of data augmentation to model generalization.

Disabled augmentations included:

* Mosaic
* Horizontal and vertical flipping
* HSV transformations
* Translation
* Scaling
* Rotation
* Shearing
* Perspective transformation
* MixUp
* Copy-Paste

### Experiment 4: Training From Scratch

The YOLOv8n architecture was trained without pretrained weights. This experiment evaluates the importance of transfer learning and pretrained initialization.

---

## Results

The experiments were evaluated using standard object detection metrics:

* Precision
* Recall
* mAP50
* mAP50-95

The results provide insight into:

* The impact of model size
* The effectiveness of data augmentation
* The importance of pretrained weights
* Performance differences between training strategies

---

## Project Structure

```text
fire-smoke-detection-yolo/

├── configs/
│   └── data.yaml
│
├── notebooks/
│   └── fire_smoke_detection_yolov8.ipynb
│
├── src/
│   ├── dataset_audit.py
│   ├── visualize_samples.py
│   ├── box_size_analysis.py
│   └── eda_plots.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

## Technologies Used

* Python
* PyTorch
* YOLOv8
* Ultralytics
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Seaborn

## Author

Ecem Nur Bilgi

TED University – Computer Engineering

ADA447 Deep Learning Project
