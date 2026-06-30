import streamlit as st


def hide_default_nav():
    st.set_option("client.showSidebarNavigation", False)


def _derive_active_step(current_step=None):
    session_step = st.session_state.get("current_step")
    if session_step is not None:
        return min(max(int(session_step), 1), 4)

    if current_step is not None:
        return min(max(current_step, 1), 4)

    if st.session_state.get("step_complete_4", False):
        return 5
    if st.session_state.get("step_complete_3", False):
        return 4
    if st.session_state.get("step_complete_2", False):
        return 3
    if st.session_state.get("step_complete_1", False):
        return 2
    return 1


def render_workflow_nav(current_step=None):
    st.sidebar.title("Workflow")

    if st.sidebar.button("🏠 Home", use_container_width=True, key="nav_home"):
        st.switch_page("home.py")

    steps = [
        ("1. Upload & split", "pages/01_upload_split.py", 1),
        ("2. Preprocess", "pages/02_preprocess_cv.py", 2),
        ("3. Tune model", "pages/03_model_tuning.py", 3),
        ("4. Evaluate", "pages/04_evaluate.py", 4),
    ]

    active_step = _derive_active_step(current_step)
    for label, target, step in steps:
        step_done = False
        if step == 1:
            step_done = st.session_state.get("step_complete_1", False)
        elif step == 2:
            step_done = st.session_state.get("step_complete_2", False)
        elif step == 3:
            step_done = st.session_state.get("step_complete_3", False)
        elif step == 4:
            step_done = st.session_state.get("step_complete_4", False)

        if step_done:
            display_label = f"✅ {label}"
            disabled = True
        elif step == active_step:
            display_label = f"▶ {label}"
            disabled = False
        else:
            display_label = f"⏳ {label}"
            disabled = True

        if st.sidebar.button(display_label, key=f"nav_{step}", disabled=disabled, use_container_width=True):
            st.switch_page(target)
