import streamlit as st
from main import run_pipeline

st.title("ML Pipeline with Streamlit")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    with open("data.csv", "wb") as f:
        f.write(uploaded_file.read())

    st.success("File uploaded and saved!")

    if st.button("Run Pipeline"):
        accuracy, report = run_pipeline("data.csv")

        st.subheader("Model Accuracy")
        st.write(f"{accuracy:.2f}")

        st.subheader("Classification Report")
        st.json(report)

        st.success("Logs saved in `Logs/output.log`")
