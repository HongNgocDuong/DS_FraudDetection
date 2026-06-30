import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

st.set_page_config(page_title="Evaluate Model", layout="wide")

if "current_step" not in st.session_state:
    st.session_state["current_step"] = 1

current_page = Path(__file__).name
current_step = st.session_state.get("current_step", 1)
page_step = 4
allowed = page_step <= current_step
if not allowed:
    target_page = ["home.py", "pages/01_upload_split.py", "pages/02_preprocess_cv.py", "pages/03_model_tuning.py", "pages/04_evaluate.py"][current_step - 1]
    st.info("This workflow only moves forward. You are being returned to the current step.")
    st.switch_page(target_page)
    st.stop()

st.title("4. Apply Preprocessing to Test Set and Evaluate")

if not st.session_state.get("step_complete_3", False):
    st.info("Please complete Step 3 first before evaluating the model.")
    st.stop()

preprocessor = st.session_state["preprocessor"]
X_test_prepared = preprocessor.transform(st.session_state["X_test"])

predictions = st.session_state["best_model"].predict(X_test_prepared)
probabilities = st.session_state["best_model"].predict_proba(X_test_prepared)[:, 1]
y_test = st.session_state["y_test"].astype(int)

metrics = {
    "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
    "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
    "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
    "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
    "pr_auc": round(float(average_precision_score(y_test, probabilities)), 4),
}

st.session_state["step_complete_4"] = True
st.session_state["current_step"] = 5
st.success("Evaluation completed on the held-out test set.")

metric_cols = st.columns(5)
metric_items = [
    ("Accuracy", metrics["accuracy"]),
    ("Precision", metrics["precision"]),
    ("Recall", metrics["recall"]),
    ("F1 Score", metrics["f1_score"]),
    ("PR AUC", metrics["pr_auc"]),
]
for col, (label, value) in zip(metric_cols, metric_items):
    col.metric(label, f"{value:.4f}")

st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("Classification Report")
report_df = pd.DataFrame(classification_report(y_test, predictions, target_names=["Negative", "Positive"], output_dict=True)).T
st.dataframe(report_df, use_container_width=True)

st.subheader("Insights")
if metrics["recall"] < 0.7:
    st.write("- Recall is below 0.7, so the model is missing a meaningful share of positive cases.")
else:
    st.write("- Recall is strong, so the model is catching most of the positive cases.")

if metrics["precision"] < 0.5:
    st.write("- Precision is relatively low, which means false positives remain a concern.")
else:
    st.write("- Precision is healthy, so the flagged cases are more likely to be true positives.")

if metrics["pr_auc"] > 0.8:
    st.write("- PR AUC is strong, indicating the model ranks positive cases well under class imbalance.")
else:
    st.write("- PR AUC is moderate, so the model could benefit from more tuning or additional feature engineering.")
