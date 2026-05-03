import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# LOAD MODEL & SCALER
# -----------------------------
model = load_model("hotel_ann_model.keras")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Hotel Booking Prediction",
    page_icon="🏨",
    layout="wide"
)

# -----------------------------
# TITLE
# -----------------------------
st.markdown("<h1 style='text-align:center;'>🏨 Hotel Booking Cancellation Predictor</h1>", unsafe_allow_html=True)

st.markdown("### Enter Booking Details")

# -----------------------------
# INPUT SECTION (2 COLUMNS)
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    no_of_adults = st.number_input("No. of Adults", min_value=0, max_value=10, value=2)
    no_of_children = st.number_input("No. of Children", min_value=0, max_value=10, value=0)
    no_of_weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=10, value=1)
    no_of_week_nights = st.number_input("Week Nights", min_value=0, max_value=20, value=2)
    lead_time = st.number_input("Lead Time", min_value=0, value=50)

with col2:
    arrival_month = st.selectbox("Arrival Month", list(range(1,13)))
    meal_plan = st.selectbox("Meal Plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
    room_type = st.selectbox("Room Type", ["Room_Type 1","Room_Type 2","Room_Type 3","Room_Type 4","Room_Type 5","Room_Type 6","Room_Type 7"])
    market_segment = st.selectbox("Market Segment", ["Online","Offline","Corporate","Aviation","Complementary"])
    avg_price = st.number_input("Avg Price Per Room", min_value=0.0, value=100.0)

# -----------------------------
# MANUAL LABEL ENCODING (IMPORTANT)
# -----------------------------
meal_map = {"Meal Plan 1":0, "Meal Plan 2":1, "Meal Plan 3":2, "Not Selected":3}
room_map = {
    "Room_Type 1":0, "Room_Type 2":1, "Room_Type 3":2,
    "Room_Type 4":3, "Room_Type 5":4, "Room_Type 6":5, "Room_Type 7":6
}
market_map = {"Online":0, "Offline":1, "Corporate":2, "Aviation":3, "Complementary":4}

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("🔍 Predict Booking Status"):

    # Create input array (MATCH TRAINING FEATURES COUNT EXACTLY)
    input_data = np.array([[
        no_of_adults,
        no_of_children,
        no_of_weekend_nights,
        no_of_week_nights,
        meal_map[meal_plan],
        room_map[room_type],
        lead_time,
        arrival_month,
        market_map[market_segment],
        avg_price
    ]])

    # IMPORTANT: HANDLE FEATURE SIZE
    # Your scaler expects 14 features → pad missing ones with 0
    if input_data.shape[1] < scaler.n_features_in_:
        padding = scaler.n_features_in_ - input_data.shape[1]
        input_data = np.pad(input_data, ((0,0),(0,padding)), mode='constant')

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0][0]

    # -----------------------------
    # RESULT DISPLAY (NO SCROLL ISSUE)
    # -----------------------------
    st.markdown("---")

    if prediction > 0.5:
        st.error("❌ Booking will likely be **CANCELLED**")
        result_text = "Cancelled"
    else:
        st.success("✅ Booking will likely be **CONFIRMED**")
        result_text = "Not Cancelled"

    # -----------------------------
    # REPORT SECTION (DOWNLOADABLE)
    # -----------------------------
    st.markdown("### 📄 Prediction Report")

    report = f"""
    Hotel Booking Prediction Report
    --------------------------------
    Adults: {no_of_adults}
    Children: {no_of_children}
    Weekend Nights: {no_of_weekend_nights}
    Week Nights: {no_of_week_nights}
    Lead Time: {lead_time}
    Arrival Month: {arrival_month}
    Meal Plan: {meal_plan}
    Room Type: {room_type}
    Market Segment: {market_segment}
    Avg Price: {avg_price}

    --------------------------------
    Prediction: {result_text}
    Probability: {prediction:.2f}
    """

    st.text_area("Report Preview", report, height=300)

    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name="booking_report.txt",
        mime="text/plain"
    )