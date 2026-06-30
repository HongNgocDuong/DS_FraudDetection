import streamlit as st


def hide_default_nav():
    st.set_option("client.showSidebarNavigation", False)


def reset_workflow_state():
    keys_to_clear = [
        "df",
        "target_column",
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        "test_size",
        "random_state",
        "preprocessor",
        "cv",
        "fold_summaries",
        "X_train_prepared",
        "best_model",
        "best_params",
        "preprocess_started",
        "evaluate_started",
        "step_complete_1",
        "step_complete_2",
        "step_complete_3",
        "step_complete_4",
        "current_step",
        "latest_completed_step",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)
    st.session_state["current_step"] = 1


def _derive_active_step(current_step=None):
    if st.session_state.get("step_complete_4", False):
        return 4
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

    if st.session_state.get("latest_completed_step") == 2:
        st.sidebar.success("Step 2 complete. Tune model is now available.")
    elif st.session_state.get("latest_completed_step") == 3:
        st.sidebar.success("Step 3 complete. Evaluation is now available.")
    elif st.session_state.get("latest_completed_step") == 4:
        st.sidebar.success("Workflow complete. You can start a new upload.")
