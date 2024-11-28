# Crop Disease Detection Using Deep Learning 🌱🧪

This repository contains a deep learning-based solution for detecting crop diseases from images using transfer learning. The goal is to assist farmers in identifying crop diseases early, improving yields, and reducing economic losses. The project leverages the **ResNet50V2** architecture and the **PlantVillage dataset** for training and evaluation.

---

## 🌟 **Project Overview**
- **Objective**: To classify 39 different crop disease categories using smartphone-captured images.
- **Model**: Convolutional Neural Network (CNN) with transfer learning using **ResNet50V2**.
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
│── app.py              
│── gem.py           
├── requirements.txt          # Required Python libraries
└── README.md                 # Project documentation

⚙️ Setup & Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/crop-disease-detection.git
cd crop-disease-detection
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Download the dataset:

PlantVillage Dataset (download and place it in the data/ folder)
