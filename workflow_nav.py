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


def render_workflow_nav(current_step=None):
    st.sidebar.title("Workflow")

    if st.sidebar.button("🏠 Home", use_container_width=True, key="nav_home"):
        st.switch_page("home.py")

    steps = [
        ("1. Dataset overview", "pages/01_upload_split.py", 1),
        ("2. Preprocessing", "pages/02_preprocess_cv.py", 2),
        ("3. Tune model", "pages/03_model_tuning.py", 3),
        ("4. Evaluate", "pages/04_evaluate.py", 4),
    ]

    for label, target, step in steps:
        display_label = f"✅ {label}"
        if st.sidebar.button(display_label, key=f"nav_{step}", use_container_width=True):
            st.switch_page(target)

    st.sidebar.success("Static demo mode is enabled. Every page shows preloaded results.")
