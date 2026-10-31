from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "student_model.pkl"
DATA_PATH = ROOT / "data" / "student_performance.csv"

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="centered")
st.title("🎓 Student Performance Prediction")
st.write("Enter student details to estimate the likely result using a Random Forest model.")
st.info("This is an educational demonstration using a generated sample dataset—not a real academic assessment tool.")

if not MODEL_PATH.exists():
    st.warning("Model file not found. First run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        attendance = st.slider("Attendance (%)", 40.0, 100.0, 80.0, 0.5)
        study_hours = st.slider("Study hours per day", 0.0, 8.0, 3.0, 0.5)
        previous_score = st.slider("Previous exam score", 0, 100, 65)
    with col2:
        assignment_score = st.slider("Assignment score", 0, 100, 70)
        internal_score = st.slider("Internal assessment score", 0, 100, 68)

    submitted = st.form_submit_button("Predict Result", type="primary")

if submitted:
    input_data = pd.DataFrame([{
        "attendance": attendance,
        "study_hours": study_hours,
        "previous_score": previous_score,
        "assignment_score": assignment_score,
        "internal_score": internal_score
    }])
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    confidence = float(max(probabilities)) * 100

    if prediction == "Pass":
        st.success(f"Predicted result: {prediction} 🎉")
    else:
        st.error(f"Predicted result: {prediction}")
    st.metric("Model confidence for this prediction", f"{confidence:.1f}%")
    st.caption("Confidence is the model's estimated probability, not a guarantee.")

st.divider()
st.subheader("Dataset overview")
df = pd.read_csv(DATA_PATH)
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Pass cases", int((df["result"] == "Pass").sum()))
c3.metric("Fail cases", int((df["result"] == "Fail").sum()))
st.bar_chart(df["result"].value_counts())
