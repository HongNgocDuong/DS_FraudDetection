import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from workflow_nav import hide_default_nav, render_workflow_nav

st.set_page_config(page_title="Model Tuning", layout="wide")
hide_default_nav()
render_workflow_nav(3)

if "current_step" not in st.session_state:
    st.session_state["current_step"] = 1

current_page = Path(__file__).name
current_step = st.session_state.get("current_step", 1)
page_step = 3
allowed = page_step <= current_step
if not allowed:
    target_page = ["home.py", "pages/01_upload_split.py", "pages/02_preprocess_cv.py", "pages/03_model_tuning.py", "pages/04_evaluate.py"][current_step - 1]
    st.info("This workflow only moves forward. You are being returned to the current step.")
    st.switch_page(target_page)
    st.stop()

st.title("3. Train and Tune the Random Forest Model")

if not st.session_state.get("step_complete_2", False):
    st.info("Please complete Step 2 first before tuning the model.")
    st.stop()

st.write("Random Forest will be tuned using stratified cross-validation with SMOTE applied inside each fold.")

if st.button("Run hyperparameter tuning", use_container_width=True):
    if "X_train_prepared" not in st.session_state:
        preprocessor = st.session_state.get("preprocessor")
        if preprocessor is None:
            st.error("Preprocessing state is missing. Please return to Step 2 and complete it again.")
            st.stop()
        st.session_state["X_train_prepared"] = preprocessor.transform(st.session_state["X_train"])

    pipeline = ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=st.session_state.get("random_state", 42))),
            (
                "classifier",
                RandomForestClassifier(random_state=st.session_state.get("random_state", 42)),
            ),
        ]
    )

    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [3, 5, 7],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__class_weight": [None, "balanced", "balanced_subsample"],
    }

    tuner = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=12,
        scoring="average_precision",
        cv=st.session_state["cv"],
        n_jobs=-1,
        random_state=st.session_state.get("random_state", 42),
        verbose=0,
    )

    tuner.fit(st.session_state["X_train_prepared"], st.session_state["y_train"].astype(int))
    st.session_state["best_model"] = tuner.best_estimator_
    st.session_state["best_params"] = tuner.best_params_
    st.session_state["step_complete_3"] = True
    st.session_state["step_complete_4"] = False
    st.session_state["current_step"] = 4
    st.success("Hyperparameter tuning completed.")
    st.json(st.session_state["best_params"])
else:
    st.info("Click the button to tune the model.")
