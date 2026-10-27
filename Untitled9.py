#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺"
)

# Load trained model and feature selector
model = joblib.load("logistic.pkl")
selector = joblib.load("diabetes_selector.pkl")

# Title
st.title("🩺 Diabetes Prediction")

st.write("Enter the health details below:")

# Input features
pregnancies = st.number_input(
    "Pregnancies", min_value=0, value=1
)

glucose = st.number_input(
    "Glucose", min_value=0.0, value=120.0
)

blood_pressure = st.number_input(
    "Blood Pressure", min_value=0.0, value=70.0
)

skin_thickness = st.number_input(
    "Skin Thickness", min_value=0.0, value=20.0
)

insulin = st.number_input(
    "Insulin", min_value=0.0, value=80.0
)

bmi = st.number_input(
    "BMI", min_value=0.0, value=30.0
)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    value=0.5
)

age = st.number_input(
    "Age", min_value=1, value=30
)

# Prediction
if st.button("Predict Diabetes"):

    data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }])

    # Select the same 5 features used during training
    selected_data = selector.transform(data)

    # Make prediction
    prediction = model.predict(selected_data)[0]

    probability = model.predict_proba(selected_data)[0][1]

    if prediction == 1:
        st.error("⚠️ Prediction: Diabetes")
    else:
        st.success("✅ Prediction: No Diabetes")

    st.write(
        f"Model-estimated diabetes probability: {probability:.2%}"
    )

    st.caption(
        "Educational project only. This prediction is not a medical diagnosis."
    )


# In[ ]:




