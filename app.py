# save this as app.py

import streamlit as st
import pandas as pd
import joblib

# Load the saved model and scaler
model_data = joblib.load('random_forest_breast_cancer_model.pkl')
scaler = model_data['scaler']
model = model_data['model']

st.title("Breast Cancer Prediction Web App")

st.write("""
Enter the tumor features below to predict whether the tumor is Benign or Malignant.
""")

# List of feature names
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points',
    'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error',
    'perimeter error', 'area error', 'smoothness error', 'compactness error',
    'concavity error', 'concave points error', 'symmetry error',
    'fractal dimension error', 'worst radius', 'worst texture',
    'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness',
    'worst concavity', 'worst concave points', 'worst symmetry',
    'worst fractal dimension'
]

# Create input widgets dynamically
user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", format="%.5f")
    user_input.append(value)

# Convert input to DataFrame
input_df = pd.DataFrame([user_input], columns=feature_names)

if st.button("Predict"):
    # Scale the input
    input_scaled = scaler.transform(input_df.values)
    # Predict
    prediction = model.predict(input_scaled)[0]
    result = "Malignant (Cancerous)" if prediction == 1 else "Benign (Non-cancerous)"
    st.success(f"Prediction: **{result}**")
