import streamlit as st
from workflow_nav import hide_default_nav, render_workflow_nav
from static_demo import ensure_static_demo_state

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")
hide_default_nav()
render_workflow_nav(1)

st.title("Fraud Detection Demo")
st.write("This version of the app is a static showcase that uses the bundled fraud sample dataset and displays results on every page.")

ensure_static_demo_state()

if "static_demo_error" in st.session_state:
    st.error(st.session_state["static_demo_error"])
    st.stop()

st.success(f"Loaded sample dataset: {st.session_state['dataset_name']}")
st.caption("No uploads or recalculation are required. The results below are preloaded for demonstration purposes.")

st.markdown("""
### Demo overview
1. Review the uploaded sample dataset and the selected target column.
2. Inspect the preprocessing plan and cross-validation setup.
3. View the tuned Random Forest hyperparameters and training summary.
4. Review the final evaluation metrics, confusion matrix, and report.
""")

steps = [
    ("1. Dataset overview", st.session_state.get("step_complete_1", False)),
    ("2. Preprocessing summary", st.session_state.get("step_complete_2", False)),
    ("3. Model tuning results", st.session_state.get("step_complete_3", False)),
    ("4. Evaluation results", st.session_state.get("step_complete_4", False)),
]

for label, done in steps:
    status = "✅ Completed" if done else "✅ Ready"
    st.write(f"{status} — {label}")

st.subheader("Sample dataset preview")
st.dataframe(st.session_state["df"].head(), use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Rows", st.session_state["dataset_shape"][0])
col2.metric("Columns", st.session_state["dataset_shape"][1])
col3.metric("Target column", st.session_state["target_column"])
