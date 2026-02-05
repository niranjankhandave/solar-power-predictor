import streamlit as st
import numpy as np
import joblib
from datetime import datetime

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Solar DC Power Predictor",
    layout="wide"
)

st.title("🔆 Solar DC Power Prediction App")

# --------------------------------------------------
# Load pretrained model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("prediction model.pkl")  # make sure filename matches

model = load_model()

st.subheader("⚡ Enter Weather & Irradiation Details")

# --------------------------------------------------
# Input layout
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    date_time = st.datetime_input(
        "Date & Time",
        value=datetime.now()
    )

with col2:
    ambient_temp = st.number_input(
        "Ambient Temperature (°C)",
        value=25.0
    )

with col3:
    module_temp = st.number_input(
        "Module Temperature (°C)",
        value=35.0
    )

irradiation = st.number_input(
    "Irradiation (W/m²)",
    min_value=0.0,
    value=800.0
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict DC Power"):
    # Convert datetime to timestamp (numeric)
    date_time_ts = int(date_time.timestamp())

    # Model was trained on:
    # [DATE_TIME, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION]
    input_data = np.array([[
        date_time_ts,
        ambient_temp,
        module_temp,
        irradiation
    ]])

    prediction = model.predict(input_data)[0]

    st.success("Prediction Complete ✅")
    st.metric(
        label="🔋 Predicted DC Power",
        value=f"{prediction:.2f} kW"
    )