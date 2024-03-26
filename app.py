# app.py
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
try:
    #st.write("Attempting to load model from 'best_model.pkl'...")
    loaded_model = joblib.load('best_model.pkl')  # Load the best model
    #st.write("Model loaded successfully!")
    #st.write("Loaded model:", loaded_model)  # Print loaded model for debugging
except Exception as e:
    st.write("Error loading model:", e)

# Streamlit interface
st.title("Heart Disease Prediction")

# Add input fields for user to input data
age = st.number_input("Enter age:")
sex = st.selectbox("Select gender:", ["Male", "Female"])
cp = st.number_input("Enter chest pain type (0-3):")
trestbps = st.number_input("Enter resting blood pressure:")
chol = st.number_input("Enter serum cholesterol level in mg/dl:")
fbs = st.selectbox("Is fasting blood sugar > 120 mg/dl?", ["No", "Yes"])
restecg = st.number_input("Enter resting electrocardiographic results (0-2):")
thalach = st.number_input("Enter maximum heart rate achieved:")
exang = st.selectbox("Is exercise induced angina present?", ["No", "Yes"])
oldpeak = st.number_input("Enter ST depression induced by exercise relative to rest:")
slope = st.number_input("Enter slope of the peak exercise ST segment (0-2):")
ca = st.number_input("Enter number of major vessels colored by fluoroscopy (0-3):")
thal = st.number_input("Enter thalassemia type (1-3):")

# Convert user input to numerical values
sex = 0 if sex == "Male" else 1
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# Predict button
if st.button("Predict"):
    # Make prediction using the loaded model
    if loaded_model:  # Check if loaded_model is not None
        user_input = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        try:
            prediction = loaded_model.predict(user_input)
            st.write("Prediction:", prediction[0])
        except Exception as e:
            st.write("Error making prediction:", e)
    else:
        st.write("Model not loaded or invalid, unable to make prediction.")
