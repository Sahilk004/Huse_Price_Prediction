# app.py

import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="House Price Prediction", layout="centered")
st.title("🏠 House Price Prediction App")
st.markdown("Enter the house details below:")

# Sidebar inputs
overall_qual = st.number_input("Overall Quality (1-10)", 1, 10, 7)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", 500, 5000, 1500)
garage_cars = st.number_input("Garage Capacity (Cars)", 0, 5, 2)
garage_area = st.number_input("Garage Area (sq ft)", 0, 1500, 400)
total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", 0, 3000, 800)
full_bath = st.number_input("Number of Full Bathrooms", 0, 4, 2)
year_built = st.number_input("Year Built", 1900, 2024, 2005)

# Predict button
if st.button("Predict Price"):
    input_data = np.array([[overall_qual, gr_liv_area, garage_cars,
                            garage_area, total_bsmt_sf, full_bath, year_built]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.success(f"🏡 Estimated House Price: ${prediction:,.2f}")


import joblib
import os

# Path correction
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "house_price_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

# Load the model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
