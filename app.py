import streamlit as st
import pickle
import numpy as np

@st.cache_resource
def load_scaler(path: str = 'model/scaler.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_model(path: str = 'model/ridge_cv.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

scaler = load_scaler()
model  = load_model()

st.title("CO₂ Emission Predictor ")
st.write("Enter the vehicle specs below and click **Predict** to see the estimated CO₂ emissions.")

engine_size = st.number_input(
    "Engine Size (L)", min_value=0.0, max_value=10.0, value=2.5, step=0.1
)
cylinders = st.number_input(
    "Cylinders", min_value=2, max_value=16, value=4, step=1
)
fuel_city = st.number_input(
    "Fuel Consumption City (L/100 km)", min_value=0.0, max_value=30.0, value=9.0, step=0.1
)
fuel_hwy = st.number_input(
    "Fuel Consumption Hwy (L/100 km)", min_value=0.0, max_value=20.0, value=6.0, step=0.1
)
fuel_comb_l100 = st.number_input(
    "Fuel Consumption Comb (L/100 km)", min_value=0.0, max_value=30.0, value=8.0, step=0.1
)
fuel_comb_mpg = st.number_input(
    "Fuel Consumption Comb (mpg)", min_value=5.0, max_value=100.0, value=30.0, step=0.5
)

if st.button("Predict CO₂ Emissions"):
    X = np.array([[ 
        engine_size,
        cylinders,
        fuel_city,
        fuel_hwy,
        fuel_comb_l100,
        fuel_comb_mpg
    ]])
    
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    
    st.markdown(f"## Predicted CO₂ Emissions: **{pred[0]:.2f} g/km**")

