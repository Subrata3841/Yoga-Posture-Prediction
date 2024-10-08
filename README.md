# Yoga Posture Prediction using Decision Tree, Random Forest, KNN, and CNN

This project aims to classify yoga postures using various machine learning and deep learning models such as Decision Tree, Random Forest, KNN, and CNN. We leverage image data and preprocess it to feed into these models for classification. The CNN model achieves higher accuracy compared to the other methods. 

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
  - [Decision Tree](#decision-tree)
  - [Random Forest](#random-forest)
  - [KNN](#knn)
  - [CNN](#cnn)
- [Results](#results)
- [Installation](#installation)
- [References](#references)

## Project Overview
The goal of this project is to predict the correct yoga posture from a dataset of labeled yoga images. The models explored in this project include:
- **Decision Tree (using Gini Index)**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**
- **Convolutional Neural Networks (CNN)**

We compare the accuracy of each model and visualize the results through different types of plots such as bar graphs and pie charts.

## Dataset
The dataset consists of 5 yoga postures:
- Goddess
- Downward Dog
- Tree Pose
- Plank
- Warrior 2

### Dataset Statistics
- **Total Samples:** 1081
- **Training Set Split:** 85% training, 15% validation

### Image Preprocessing
- Images are resized to (100x100) pixels.
- Labels are one-hot encoded for the CNN model.

## Model Architectures

### Decision Tree
- **Criterion:** Gini
- **Max Features:** sqrt
- **Min Samples Split:** 2
- **Accuracy:** ~55%

### Random Forest
- **Number of Trees (Estimators):** 100
- **Criterion:** Gini
- **Max Features:** sqrt
- **Min Samples Split:** 2
- **Accuracy:** ~61%

### KNN
- **Neighbors:** 5
- **Metric:** Minkowski
- **Accuracy:** ~52%

### CNN
- **Base Model:** Xception (Pre-trained)
- **Additional Layers:** Dense layers with ReLU activation and dropout for regularization.
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Accuracy:** ~95%

## Results
| Model          | Accuracy |
| -------------- | -------- |
| Decision Tree  | 55%      |
| Random Forest  | 61%      |
| KNN            | 52%      |
| CNN            | 95%      |

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/yoga-posture-prediction.git
   cd yoga-posture-prediction
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have access to Google Colab for running the notebook.

## References
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Xception Model](https://keras.io/api/applications/xception/)

