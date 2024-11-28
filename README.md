# Crop Disease Detection Using Deep Learning 🌱🧪

This repository contains a deep learning-based solution for detecting crop diseases from images using transfer learning. The goal is to assist farmers in identifying crop diseases early, improving yields, and reducing economic losses. The project leverages the **VGG16** architecture and the **PlantVillage dataset** for training and evaluation.

---

## 🌟 **Project Overview**
- **Objective**: To classify 39 different crop disease categories using smartphone-captured images.
- **Model**: Convolutional Neural Network (CNN) with transfer learning using **VGG16** and **ResNet50V2**.
- **Dataset**: The publicly available [PlantVillage dataset](https://www.kaggle.com/emmarex/plantdisease) from Kaggle.
- **Deployment**: The model is intended to be deployed as a mobile/web app for real-time disease detection.

---

## 🚀 **Key Features**
- **Real-Time Disease Detection**: Upload an image and get instant disease diagnosis.
- **Transfer Learning**: Utilizes pretrained VGG16 for feature extraction and classification.
- **Data Augmentation**: Enhances training with transformations like rotation, flipping, and zooming.
- **Regularization**: Implements early stopping and dropout to prevent overfitting.
- **Evaluation Metrics**: Achieved 94% test accuracy, with a precision of 92.9% and an F1-score of 92.5%.

---

## 📂 **Project Structure**
```plaintext
├── data/
│   ├── train/                # Training images
│   ├── validation/           # Validation images
│   └── test/                 # Test images
├── crop_disease_model.h5 # Saved trained model
├── tansfer_learning.ipynb # Jupyter notebook for training
├── static/
│   |── css      # Sample images for testing
    └── uploads
├── templates/
│   |── index.html  
│── model.py              # Model architecture and training code
│── utils.py              # Utility functions for preprocessing
│── predict.py            # Script for making predictions
├── requirements.txt          # Required Python libraries
└── README.md                 # Project documentation

