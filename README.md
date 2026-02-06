# Medical Image Analysis Framework

This repository contains the implementation of a unified framework for medical image analysis using retinal OCT and thyroid ultrasound images. Modules include:

- Preprocessing and augmentation
- Segmentation (CRP-URNet)
- Handcrafted and deep feature extraction
- Hybrid feature fusion
- Classification (WideNet)
- Training and evaluation

## Requirements
- Python 3.8+
- TensorFlow 2.x
- Keras
- OpenCV
- scikit-image
- numpy
- scikit-learn

## Usage
1. Preprocess dataset using `preprocessing.py`.
2. Train segmentation model using `segmentation.py`.
3. Extract features using `feature_extraction.py`.
4. Fuse features with `feature_fusion.py`.
5. Train classifier with `classification.py` + `training_evaluation.py`.

Datasets:
- [OCT Dataset](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)
- [DDTI Thyroid Ultrasound](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images)
- [TN5000 Thyroid Ultrasound](https://figshare.com/s/cb6a67f17c04b29e7edd)
