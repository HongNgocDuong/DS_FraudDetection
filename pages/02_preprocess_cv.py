import pandas as pd
import streamlit as st
from workflow_nav import hide_default_nav, render_workflow_nav
from static_demo import ensure_static_demo_state

st.set_page_config(page_title="Preprocessing Summary", layout="wide")
hide_default_nav()
render_workflow_nav(2)

ensure_static_demo_state()

st.title("2. Preprocessing Summary")
st.write("This page shows the preprocessing setup for the bundled sample dataset. No transformations are executed again.")

summary = st.session_state["preprocess_summary"]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Cross-validation folds", summary["folds"])
col2.metric("Scaled columns", ", ".join(summary["scaled_columns"]) if summary["scaled_columns"] else "None")
col3.metric("Outlier capping", "Enabled" if summary["outlier_capping"] else "Disabled")
col4.metric("SMOTE", "Enabled" if summary["smote"] else "Disabled")

st.subheader("Configured steps")
steps_df = pd.DataFrame(
    [
        {"Step": "Stratified K-Fold split", "Setting": "5 folds"},
        {"Step": "Outlier treatment", "Setting": "3-sigma capping on numeric features"},
        {"Step": "Scaling", "Setting": "RobustScaler applied to selected numeric columns"},
        {"Step": "Class imbalance handling", "Setting": "SMOTE applied inside each fold"},
    ]
)
st.dataframe(steps_df, use_container_width=True)

st.caption(f"Feature count included in the demo: {summary['feature_count']}")
