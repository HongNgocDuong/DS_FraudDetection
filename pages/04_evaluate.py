import pandas as pd
import streamlit as st
from workflow_nav import hide_default_nav, render_workflow_nav
from static_demo import ensure_static_demo_state

st.set_page_config(page_title="Evaluation Results", layout="wide")
hide_default_nav()
render_workflow_nav(4)

ensure_static_demo_state()

st.title("4. Evaluation Results")
st.write("The evaluation metrics below are precomputed for the sample dataset. This page does not run any model inference.")

summary = st.session_state["evaluation_summary"]
metric_cols = st.columns(5)
metric_items = [
    ("Accuracy", summary["accuracy"]),
    ("Precision", summary["precision"]),
    ("Recall", summary["recall"]),
    ("F1 Score", summary["f1_score"]),
    ("PR AUC", summary["pr_auc"]),
]
for col, (label, value) in zip(metric_cols, metric_items):
    col.metric(label, f"{value:.4f}")

st.subheader("Confusion Matrix")
confusion_df = pd.DataFrame(
    summary["confusion_matrix"],
    columns=["Predicted Negative", "Predicted Positive"],
    index=["Actual Negative", "Actual Positive"],
)
st.dataframe(confusion_df, use_container_width=True)

st.subheader("Classification Report")
report_df = pd.DataFrame(
    {
        "precision": [0.95, 0.94],
        "recall": [0.97, 0.92],
        "f1-score": [0.96, 0.93],
        "support": [188, 112],
    },
    index=["Negative", "Positive"],
)
st.dataframe(report_df, use_container_width=True)

st.caption("These outputs are intentionally fixed so the app can be reviewed as a polished presentation demo.")
