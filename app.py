import streamlit as st
import numpy as np
import pickle
import sklearn

# Show sklearn version (optional)
# st.write("Scikit-learn version:", sklearn._version_)

# Load trained model & preprocessor
dtr = pickle.load(open('dtr.pkl', 'rb'))
Models = pickle.load(open('preprocessor.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Crop Yield Prediction", layout="centered")

# Title
st.markdown(
    "<h1 style='text-align:center; color:green;'>🌾 Crop Yield Prediction Per Country</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

# Input section
st.markdown(
    "<h3 style='color:red;'>Input All Features Here</h3>",
    unsafe_allow_html=True
)

# Inputs (same as your HTML form)
Year = st.number_input("Year", value=2013, step=1)
average_rain_fall_mm_per_year = st.number_input(
    "Average Rainfall (mm per year)", step=0.1
)
pesticides_tonnes = st.number_input(
    "Pesticides (tonnes)", step=0.1
)
avg_temp = st.number_input(
    "Average Temperature (°C)", step=0.1
)
Area = st.text_input("Area (Country / Region)")
Item = st.text_input("Item (Crop Name)")

# Predict button
if st.button("🔴 Predict", use_container_width=True):
    # Prepare input features
    features = np.array(
        [[
            Year,
            average_rain_fall_mm_per_year,
            pesticides_tonnes,
            avg_temp,
            Area,
            Item
        ]],
        dtype=object
    )

    # Transform features
    transformed_features = Models.transform(features)

    # Predict
    prediction = dtr.predict(transformed_features)

    # Output
    st.markdown("---")
    st.success(f"🌱 *Predicted Yield:* {prediction[0]:.2f}")