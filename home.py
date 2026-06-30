import streamlit as st
import pandas as pd

st.set_page_config(page_title="Fraud Detection Pipeline", layout="wide")
st.title("Fraud Detection ML Pipeline")
st.write("This multi-page app walks through the full fraud detection workflow from data upload to final evaluation.")

st.markdown("""
### Pipeline Overview
1. Upload your dataset and create a train/test split.
2. Prepare the training data with stratified cross-validation, outlier capping, RobustScaler, and SMOTE.
3. Train and tune a Random Forest model across folds.
4. Apply the same preprocessing to the test set and evaluate the final model.
""")

st.info("Complete each step in order. Later pages stay locked until the previous step is finished.")

steps = [
    ("1. Upload and split", st.session_state.get("step_complete_1", False)),
    ("2. Preprocess and cross-validation", st.session_state.get("step_complete_2", False)),
    ("3. Model tuning", st.session_state.get("step_complete_3", False)),
    ("4. Evaluation", st.session_state.get("step_complete_4", False)),
]

for label, done in steps:
    status = "✅" if done else "⏳"
    st.write(f"{status} {label}")

if "df" in st.session_state:
    st.success("Dataset loaded and ready for the next step.")
    st.dataframe(st.session_state["df"].head(), use_container_width=True)
else:
    st.info("Upload a dataset from the first page to begin.")
