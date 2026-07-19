import streamlit as st
from workflow_nav import hide_default_nav, render_workflow_nav
from static_demo import ensure_static_demo_state

st.set_page_config(page_title="Dataset Overview", layout="wide")
hide_default_nav()
render_workflow_nav(1)

ensure_static_demo_state()

st.title("1. Dataset Overview")
st.write("The fraud sample dataset is already loaded for this demo. No upload or split is required.")

st.success(f"Loaded sample dataset: {st.session_state['dataset_name']}")

summary = st.session_state["split_summary"]
col1, col2, col3 = st.columns(3)
col1.metric("Train rows", summary["train_rows"])
col2.metric("Test rows", summary["test_rows"])
col3.metric("Target column", summary["target_column"])

st.subheader("Sample rows")
st.dataframe(st.session_state["df"].head(), use_container_width=True)

st.caption("This page is intentionally static and displays the bundled dataset values directly.")
