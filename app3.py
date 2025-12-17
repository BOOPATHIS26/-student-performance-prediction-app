import streamlit as st
import numpy as np
import joblib

# -----------------------------
# LOAD MODEL & SCALER
# -----------------------------
model = joblib.load("student_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# APP TITLE
# -----------------------------
st.title("🎓 Student Performance Prediction App")
st.write("Predict student total score based on study and attendance details")

# -----------------------------
# INPUT FIELDS
# -----------------------------
study_hours = st.number_input(
    "Weekly Self Study Hours",
    min_value=0.0,
    max_value=50.0,
    step=0.5
)

attendance = st.number_input(
    "Attendance Percentage",
    min_value=0.0,
    max_value=100.0,
    step=1.0
)

participation = st.number_input(
    "Class Participation (0–10)",
    min_value=0.0,
    max_value=10.0,
    step=0.1
)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Total Score"):
    
    input_data = np.array([[study_hours, attendance, participation]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    st.success(f"📊 Predicted Total Score: {round(prediction, 2)}")

st.write("Made with ❤️ by BHUVI")
