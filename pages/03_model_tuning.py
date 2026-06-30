import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Model Tuning", layout="wide")
st.title("3. Train and Tune the Random Forest Model")

if "X_train_prepared" not in st.session_state or "y_train" not in st.session_state:
    st.info("Complete the preprocessing step first.")
    st.stop()

st.write("Random Forest will be tuned using stratified cross-validation with SMOTE applied inside each fold.")

if st.button("Run hyperparameter tuning"):
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
    st.success("Hyperparameter tuning completed.")
    st.json(st.session_state["best_params"])
else:
    st.info("Click the button to tune the model.")
