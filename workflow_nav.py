import streamlit as st


def hide_default_nav():
    st.set_option("client.showSidebarNavigation", False)


def render_workflow_nav(current_step: int):
    st.sidebar.title("Workflow")

    if st.sidebar.button("🏠 Home", use_container_width=True, key="nav_home"):
        st.switch_page("home.py")

    steps = [
        ("1. Upload & split", "pages/01_upload_split.py", 1),
        ("2. Preprocess", "pages/02_preprocess_cv.py", 2),
        ("3. Tune model", "pages/03_model_tuning.py", 3),
        ("4. Evaluate", "pages/04_evaluate.py", 4),
    ]

    active_step = min(max(current_step, 1), 4)
    for label, target, step in steps:
        disabled = step != active_step
        if st.sidebar.button(label, key=f"nav_{step}", disabled=disabled, use_container_width=True):
            st.switch_page(target)
