import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from workflow_nav import hide_default_nav, render_workflow_nav

st.set_page_config(page_title="Preprocess & CV", layout="wide")
hide_default_nav()
render_workflow_nav(2)

if "current_step" not in st.session_state:
    st.session_state["current_step"] = 1

current_page = Path(__file__).name
current_step = st.session_state.get("current_step", 1)
page_step = 2
allowed = page_step <= current_step
if not allowed:
    target_page = ["home.py", "pages/01_upload_split.py", "pages/02_preprocess_cv.py", "pages/03_model_tuning.py", "pages/04_evaluate.py"][current_step - 1]
    st.info("This workflow only moves forward. You are being returned to the current step.")
    st.switch_page(target_page)
    st.stop()

st.title("2. Stratified K-Fold Preprocessing")

if not st.session_state.get("step_complete_1", False):
    st.info("Please complete Step 1 first by creating a train/test split.")
    st.stop()

st.write("The workflow below creates stratified folds first and then applies preprocessing inside each fold:")
st.write("- stratified k-fold split on the training set")
st.write("- 3-sigma capping for numeric features")
st.write("- RobustScaler for Amount and Time")

if st.button("Start preprocessing", use_container_width=True):
    st.session_state["preprocess_started"] = True

if not st.session_state.get("preprocess_started", False):
    st.info("Press the button above to start preprocessing.")
    st.stop()

class FraudPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, scale_columns=None):
        self.scale_columns = scale_columns or ["Time", "Amount"]

    def fit(self, X, y=None):
        X = X.copy()
        self.numeric_columns_ = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
        self.scaler_ = {}
        for col in self.scale_columns:
            if col in self.numeric_columns_:
                scaler = RobustScaler()
                scaler.fit(X[[col]].astype(float))
                self.scaler_[col] = scaler
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.numeric_columns_:
            if col not in X.columns:
                continue
            if not pd.api.types.is_numeric_dtype(X[col]):
                continue
            if X[col].nunique(dropna=True) <= 1:
                continue
            mean = X[col].mean()
            std = X[col].std()
            if pd.isna(std) or std == 0:
                continue
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)

        for col, scaler in self.scaler_.items():
            if col in X.columns:
                X[col] = scaler.transform(X[[col]].astype(float)).ravel()
        return X


def apply_fold_preprocessing(X_train_fold, X_val_fold, scale_columns=None):
    X_train_fold = X_train_fold.copy()
    X_val_fold = X_val_fold.copy()
    numeric_columns = [col for col in X_train_fold.columns if pd.api.types.is_numeric_dtype(X_train_fold[col])]

    for col in numeric_columns:
        if X_train_fold[col].nunique(dropna=True) <= 1:
            continue
        mean = X_train_fold[col].mean()
        std = X_train_fold[col].std()
        if pd.isna(std) or std == 0:
            continue
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        X_train_fold[col] = X_train_fold[col].clip(lower=lower_bound, upper=upper_bound)
        X_val_fold[col] = X_val_fold[col].clip(lower=lower_bound, upper=upper_bound)

    scalers = {}
    for col in scale_columns or ["Time", "Amount"]:
        if col in numeric_columns:
            scaler = RobustScaler()
            scaler.fit(X_train_fold[[col]].astype(float))
            X_train_fold[col] = scaler.transform(X_train_fold[[col]].astype(float)).ravel()
            X_val_fold[col] = scaler.transform(X_val_fold[[col]].astype(float)).ravel()
            scalers[col] = scaler

    return X_train_fold, X_val_fold, scalers


random_state = st.session_state.get("random_state", 42)
y_train = st.session_state["y_train"].astype(int)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
fold_summaries = []

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(st.session_state["X_train"], y_train), start=1):
    X_train_fold = st.session_state["X_train"].iloc[train_idx].copy()
    X_val_fold = st.session_state["X_train"].iloc[val_idx].copy()
    y_train_fold = y_train.iloc[train_idx].copy()
    y_val_fold = y_train.iloc[val_idx].copy()

    X_train_fold_proc, X_val_fold_proc, _ = apply_fold_preprocessing(X_train_fold, X_val_fold, scale_columns=["Time", "Amount"])

    fold_summaries.append(
        {
            "fold": fold_idx,
            "train_rows": len(X_train_fold),
            "val_rows": len(X_val_fold),
            "class_ratio": float(y_train_fold.mean()),
        }
    )

preprocessor = FraudPreprocessor(scale_columns=["Time", "Amount"])
preprocessor.fit(st.session_state["X_train"])
X_train_prepared = preprocessor.transform(st.session_state["X_train"])

st.session_state["preprocessor"] = preprocessor
st.session_state["cv"] = cv
st.session_state["fold_summaries"] = fold_summaries
st.session_state["X_train_prepared"] = X_train_prepared
st.session_state["step_complete_2"] = True
st.session_state["current_step"] = 3
st.session_state["preprocess_started"] = True

st.success("Preprocessing complete. Step 2 is finished and the Tune model step is now unlocked.")
st.dataframe(pd.DataFrame(fold_summaries), use_container_width=True)
