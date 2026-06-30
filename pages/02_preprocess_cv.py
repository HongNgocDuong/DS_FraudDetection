import streamlit as st
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Preprocess & CV", layout="wide")
st.title("2. Stratified K-Fold Preprocessing")

if "X_train" not in st.session_state:
    st.info("Complete the previous step to create a train/test split first.")
    st.stop()

st.write("Preprocessing steps for the training set:")
st.write("- 3-sigma capping for numeric features")
st.write("- RobustScaler for Amount and Time")
st.write("- SMOTE to handle class imbalance")
st.write("- 5-fold stratified cross-validation")

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

preprocessor = FraudPreprocessor(scale_columns=["Time", "Amount"])
X_train_prepared = preprocessor.fit_transform(st.session_state["X_train"])

st.session_state["preprocessor"] = preprocessor
st.session_state["X_train_prepared"] = X_train_prepared
st.session_state["cv"] = StratifiedKFold(n_splits=5, shuffle=True, random_state=st.session_state.get("random_state", 42))
st.session_state["smote"] = SMOTE(random_state=st.session_state.get("random_state", 42))

st.success("Training preprocessing and CV configuration prepared.")
st.dataframe(pd.DataFrame(X_train_prepared).head(), use_container_width=True)
