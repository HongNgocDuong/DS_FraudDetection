import streamlit as st
from workflow_nav import hide_default_nav, render_workflow_nav
from static_demo import ensure_static_demo_state

st.set_page_config(page_title="Model Tuning Results", layout="wide")
hide_default_nav()
render_workflow_nav(3)

ensure_static_demo_state()

st.title("3. Model Tuning Results")
st.write("The best model settings below are precomputed for the bundled sample dataset. No tuning is rerun here.")

summary = st.session_state["model_summary"]
col1, col2, col3 = st.columns(3)
col1.metric("Model", summary["model"])
col2.metric("CV score", f"{summary['cv_score']:.2f}")
col3.metric("Best parameters", len(summary["best_params"]))

st.subheader("Best hyperparameters")
st.json(summary["best_params"])

st.caption("This demo keeps the workflow visual and static while still showing the same model-selection output that would appear in a full pipeline run.")
