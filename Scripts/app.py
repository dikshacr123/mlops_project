import streamlit as st
from main import run_pipeline

st.title("ML Pipeline with Streamlit")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    file_path = "Data/data.csv"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"File uploaded and saved to {file_path}")

    if st.button("Run Pipeline"):
        try:
            accuracy, report = run_pipeline(file_path)

            st.subheader("Model Accuracy")
            st.write(f"{accuracy:.2f}")

            st.success("Pipeline executed successfully. Check logs for details.")
        except Exception as e:
            st.error("Pipeline failed. Check logs for more info.")
            st.exception(e)  # Show exception in Streamlit for debugging
