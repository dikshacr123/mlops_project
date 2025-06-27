import streamlit as st
import pandas as pd
import pickle
import os
from helper_function import log_info


# Set correct paths manually
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "Artifacts")

pipeline_path = os.path.join(ARTIFACTS_DIR, "data_processing_pipeline.pkl")
label_encoder_path = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")
model_path = os.path.join(ARTIFACTS_DIR, "trained_model.pkl")

# Load artifacts
with open(pipeline_path, 'rb') as f:
    pipeline = pickle.load(f)

with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# === Default values for unused columns ===
default_values = {
    'Marital status': 1,
    'Application mode': 17,
    'Application order': 1,
    'Course': 9254,
    'Daytime/evening attendance\t': 1,  # note the tab
    'Previous qualification': 1,  # will be overridden
    'Previous qualification (grade)': 122.0,
    'Nacionality': 1,
    "Mother's qualification": 37,
    "Father's qualification": 37,
    "Mother's occupation": 9,
    "Father's occupation": 9,
    'Admission grade': 142.5,  # will be overridden
    'Displaced': 0,
    'Educational special needs': 0,
    'Debtor': 0,
    'Tuition fees up to date': 1,
    'Gender': 1,  # will be overridden
    'Scholarship holder': 0,  # will be overridden
    'Age at enrollment': 20,  # will be overridden
    'International': 0,
    'Curricular units 1st sem (credited)': 0,
    'Curricular units 1st sem (enrolled)': 6,
    'Curricular units 1st sem (evaluations)': 6,
    'Curricular units 1st sem (approved)': 6,
    'Curricular units 1st sem (grade)': 13.0,
    'Curricular units 1st sem (without evaluations)': 0,
    'Curricular units 2nd sem (credited)': 0,
    'Curricular units 2nd sem (enrolled)': 6,
    'Curricular units 2nd sem (evaluations)': 6,
    'Curricular units 2nd sem (approved)': 6,
    'Curricular units 2nd sem (grade)': 13.0,
    'Curricular units 2nd sem (without evaluations)': 0,
    'Unemployment rate': 10.0,  # will be overridden
    'Inflation rate': 0.0,      # will be overridden
    'GDP': 1.0                  # will be overridden
}

# === Streamlit UI ===
st.title("🎓 Student Outcome Predictor")

st.markdown("Enter student details below:")

# User inputs
prev_qual = st.number_input("Previous qualification", min_value=1, max_value=100, value=1)
admission_grade = st.number_input("Admission grade", min_value=0.0, max_value=200.0, value=142.5)

gender_map = {"Male": 1, "Female": 0}
gender_label = st.selectbox("Gender", list(gender_map.keys()))
gender = gender_map[gender_label]

scholarship_map = {"Yes": 1, "No": 0}
scholarship_label = st.selectbox("Scholarship holder", list(scholarship_map.keys()))
scholarship = scholarship_map[scholarship_label]

age = st.number_input("Age at enrollment", min_value=16, max_value=70, value=20)
unemployment = st.number_input("Unemployment rate (%)", value=10.0)
inflation = st.number_input("Inflation rate (%)", value=0.0)
gdp = st.number_input("GDP growth (%)", value=1.0)

# Update the default dict with user values
default_values.update({
    'Previous qualification': prev_qual,
    'Admission grade': admission_grade,
    'Gender': gender,
    'Scholarship holder': scholarship,
    'Age at enrollment': age,
    'Unemployment rate': unemployment,
    'Inflation rate': inflation,
    'GDP': gdp
})

# === Prediction logic ===
if st.button("Predict"):
    try:
        input_df = pd.DataFrame([default_values])
        transformed_input = pipeline.transform(input_df)
        prediction_encoded = pipeline.named_steps['preprocessor'].transformers_[0][1]  # for debug only

        prediction = model.predict(transformed_input)
        predicted_label = label_encoder.inverse_transform(prediction)

        st.success(f"🎯 Predicted Outcome: **{predicted_label[0]}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
