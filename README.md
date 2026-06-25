# Medical Image Analysis Framework

This repository contains the implementation of a unified framework for medical image analysis using retinal OCT and thyroid ultrasound images. The proposed framework integrates segmentation-guided feature extraction, hybrid feature fusion, and classification for automated medical image analysis.

## Modules

- Preprocessing and augmentation
- Segmentation (CRP-URNet)
- Handcrafted feature extraction (GLCM, HOG, and LBP)
- Deep feature extraction (ResNet50)
- Hybrid feature fusion
- Classification (WideNet)
- Training and evaluation
- Cross-dataset validation

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Keras
- OpenCV
- scikit-image
- numpy
- scikit-learn

## Usage

1. Preprocess the dataset using `preprocessing.py`.
2. Train the segmentation model using `segmentation.py`.
3. Extract handcrafted and deep features using `feature_extraction.py`.
4. Fuse features using `feature_fusion.py`.
5. Train the classifier using `classification.py` and `training_evaluation.py`.
6. Perform cross-dataset validation using `cross_dataset_validation.py`.

## Cross-Dataset Validation

Additional experiments are provided to evaluate the generalization capability of the proposed framework across independent thyroid ultrasound datasets.

Experiments:

- Train on TN5000 and test on DDTI.
- Train on DDTI and test on TN5000.

These experiments assess model robustness under different imaging conditions and data distributions.

## Datasets

- [OCT Dataset](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)
- [DDTI Thyroid Ultrasound](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images)
- [TN5000 Thyroid Ultrasound](https://figshare.com/s/cb6a67f17c04b29e7edd)

## Classification Model

The WideNet classifier consists of two fully connected hidden layers containing 512 and 256 neurons, followed by a softmax output layer.

## Citation

If you use this repository, please cite:

Muhammad Ismail and Fatima Umer,

"A Unified Framework for Medical Image Segmentation and Feature Fusion-Based Classification."

## License

This repository is intended for research and educational purposes.
