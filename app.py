import streamlit as st
import numpy as np
import pickle
from datetime import datetime
import base64

# Load trained model
with open("Airline_price_pred_model.pkl", "rb") as file:
    model = pickle.load(file)

# --- Add background image ---
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    page_bg_img = f"""
    <style>
    /* Background setup */
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }}

    /* Hide the default Streamlit header background */
    [data-testid="stHeader"] {{
        background: none;
    }}

    /* Form container - right side (no black box) */
    .main-container {{
        position: absolute;
        top: 50%;
        right: 6%;
        transform: translateY(-50%);
        padding: 2rem 3rem;
        border-radius: 18px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        width: 420px;
        background: rgba(0, 0, 0, 0.25);
    }}

    h1 {{
        color: #ffffff;
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.6);
    }}
    p {{
        color: #f0f0f0;
        text-align: center;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.6);
    }}

    /* Input label colors */
    label {{
        color: #ffffff !important;
        font-weight: 600 !important;
        text-shadow: 0px 0px 2px rgba(0,0,0,0.7);
    }}

    /* Predict button custom style */
    div.stButton > button:first-child {{
        background-color: #111;
        color: white;
        font-weight: 700;
        border: 2px solid #fff;
        padding: 0.6rem 2rem;
        border-radius: 10px;
        transition: all 0.3s ease-in-out;
    }}
    div.stButton > button:first-child:hover {{
        background-color: #f7f7f7;
        color: #000;
        border-color: #000;
        transform: scale(1.05);
    }}

    /* Footer styling */
    .footer {{
        text-align: center;
        position: absolute;
        bottom: 20px;
        width: 100%;
        color: white;
        font-size: 1rem;
        font-weight: 600;
        text-shadow: 1px 1px 2px #000;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# ✅ Use relative path for cloud deployment
add_bg_from_local("assets/pexels-ahmedmuntasir-912050.jpg")

# --- Main app content ---
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown("<h1>✈️ Flight Price Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p>Enter your flight details below to estimate ticket price.</p>", unsafe_allow_html=True)

# Inputs
airline = st.selectbox("Airline", [
    'Air India', 'GoAir', 'IndiGo', 'Jet Airways',
    'Multiple carriers', 'SpiceJet', 'Vistara'
])
source = st.selectbox("Source", ['Chennai', 'Delhi', 'Kolkata', 'Mumbai'])
destination = st.selectbox("Destination", [
    'Cochin', 'Delhi', 'Hyderabad', 'Kolkata',
    'New Delhi', 'Banglore', 'Chennai', 'Mumbai'
])
total_stops = st.selectbox("Total Stops", ['non-stop', '1 stop', '2 stops', '3 stops'])
journey_date = st.date_input("Date of Journey", min_value=datetime.today())
dep_time = st.time_input("Departure Time")
arr_time = st.time_input("Arrival Time")

# --- Preprocessing ---
def preprocess():
    journey_day = journey_date.day
    journey_month = journey_date.month
    dep_hour = dep_time.hour
    dep_min = dep_time.minute
    arr_hour = arr_time.hour
    arr_min = arr_time.minute

    dep_dt = datetime.combine(datetime.today(), dep_time)
    arr_dt = datetime.combine(datetime.today(), arr_time)
    duration = abs((arr_dt - dep_dt).total_seconds()) / 60
    duration_hour = int(duration // 60)
    duration_min = int(duration % 60)

    stop_map = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3}
    stops = stop_map[total_stops]

    airline_ohe = [1 if airline == x else 0 for x in
                   ['Air India', 'GoAir', 'IndiGo', 'Jet Airways',
                    'Multiple carriers', 'SpiceJet', 'Vistara']]
    source_ohe = [1 if source == x else 0 for x in ['Chennai', 'Delhi', 'Kolkata', 'Mumbai']]
    destination_ohe = [1 if destination == x else 0 for x in
                       ['Cochin', 'Delhi', 'Hyderabad', 'Kolkata',
                        'New Delhi', 'Banglore', 'Chennai', 'Mumbai']]

    final_input = [
        stops, journey_day, journey_month,
        dep_hour, dep_min, arr_hour, arr_min,
        duration_hour, duration_min
    ] + airline_ohe + source_ohe + destination_ohe

    return np.array([final_input])

# --- Prediction ---
if st.button("Predict Price"):
    input_data = preprocess()
    if len(input_data[0]) != 28:
        st.error(f"⚠️ Feature mismatch: Got {len(input_data[0])}, expected 28.")
    else:
        prediction = model.predict(input_data)
        st.success(f"💰 Estimated Flight Price: ₹{round(prediction[0], 2)}")

st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("<div class='footer'>🧠 Model built by <b>Abhishek</b> | Accuracy: <b>85%+</b></div>", unsafe_allow_html=True)
