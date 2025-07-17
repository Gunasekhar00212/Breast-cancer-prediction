# Breast Cancer Prediction – A Machine Learning Based Approach

This project aims to build a machine learning model that can predict whether a breast tumor is **benign** or **malignant** using the **Wisconsin Breast Cancer Diagnostic dataset**.

## 📌 Project Overview

Breast cancer is one of the most common and life-threatening diseases affecting women worldwide. Early detection plays a crucial role in improving survival rates. In this project, we use supervised learning algorithms to build a reliable classification model for breast cancer diagnosis.

## 🔍 Problem Statement

Given a set of features computed from a breast mass (like radius, texture, perimeter, area, etc.), predict whether the tumor is **malignant** or **benign**.

## 🧠 Algorithms Used

We experimented with the following machine learning models:

- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest ✅ *(Best Model – 94% Accuracy)*
- Naive Bayes

## 📊 Dataset

- **Name**: Wisconsin Breast Cancer Diagnostic Dataset  
- **Source**: UCI Machine Learning Repository  
- **Features**: 30 numeric features + diagnosis label (M/B)  
- **Target**: Diagnosis (Malignant = M, Benign = B)

## 🔧 Technologies Used

- Python  
- Scikit-learn  
- Pandas  
- Matplotlib & Seaborn (for visualization)  
- Jupyter Notebook / Google Colab

## 📈 Evaluation Metric

- Accuracy  
- Confusion Matrix  
- Precision, Recall, F1 Score

## 🏆 Best Model

The **Random Forest Classifier** gave the best performance with **94% accuracy**.

## 🚀 How to Run the Project

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/breast-cancer-prediction.git
   cd breast-cancer-prediction
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt

3. Run the notebook

To run the project, open the      breast_cancer_prediction.ipynb file in Jupyter Notebook or Google Colab, and execute all the cells sequentially. This will load the dataset, preprocess the data, train the machine learning models, and evaluate their performance.
