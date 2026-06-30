import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("Fraud Detection App")
st.write(
    "This workflow follows the requested sequence: split into train/test, preprocess the training data with outlier capping and RobustScaler, apply SMOTE during cross-validation, tune a Random Forest, retrain with the best hyperparameters, and evaluate on the test set."
)


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)


def encode_target(series):
    clean = series.astype(str).str.strip()
    unique_values = sorted(clean.dropna().unique())
    if len(unique_values) != 2:
        raise ValueError("The target column must contain exactly two classes for binary classification.")
    mapping = {unique_values[0]: 0, unique_values[1]: 1}
    return clean.map(mapping).astype(int)


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


def run_pipeline(df, target_column, test_size, random_state=42):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the uploaded data.")

    X = df.drop(columns=[target_column]).copy()
    y = encode_target(df[target_column])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    preprocessor = FraudPreprocessor(scale_columns=["Time", "Amount"])

    pipeline = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=random_state)),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=random_state,
                ),
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
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        n_jobs=-1,
        random_state=random_state,
        verbose=0,
    )
    tuner.fit(X_train, y_train)

    best_params = tuner.best_params_
    final_model = tuner.best_estimator_
    predictions = final_model.predict(X_test)
    probabilities = final_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y_test, probabilities)), 4),
    }

    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=["Negative", "Positive"], output_dict=True)

    feature_names = X_train.columns.tolist()
    importances = pd.DataFrame(
        {"feature": feature_names, "importance": final_model.named_steps["classifier"].feature_importances_}
    ).sort_values(by="importance", ascending=False).head(15)

    insights = []
    if metrics["recall"] < 0.7:
        insights.append("Recall is below 0.7, so the model is missing a meaningful share of positive cases.")
    else:
        insights.append("Recall is strong, so the model is catching most of the positive cases.")

    if metrics["precision"] < 0.5:
        insights.append("Precision is relatively low, which means false positives remain a concern.")
    else:
        insights.append("Precision is healthy, so the flagged cases are more likely to be true positives.")

    if metrics["pr_auc"] > 0.8:
        insights.append("The PR AUC is strong, indicating the model ranks positive cases well under class imbalance.")
    else:
        insights.append("The PR AUC is moderate, so the model could benefit from more tuning or additional feature engineering.")

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "feature_importance": importances,
        "best_params": best_params,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "insights": insights,
        "y_test": y_test,
        "predictions": predictions,
        "probabilities": probabilities,
    }


uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    with st.spinner("Loading data..."):
        df = load_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    target_options = [col for col in df.columns if col.lower() in {"class", "label", "target", "is_fraud", "fraud"}]
    if not target_options:
        target_options = df.columns.tolist()
    target_column = st.selectbox("Select the target column", target_options)

    col1, col2 = st.columns(2)
    test_size = col1.slider("Test set size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    random_state = col2.number_input("Random state", min_value=1, max_value=1000, value=42)

    if st.button("Run full fraud-detection workflow"):
        try:
            with st.spinner("Splitting data, preprocessing the training set, tuning XGBoost, and evaluating on the test set..."):
                result = run_pipeline(df, target_column, float(test_size), int(random_state))

            st.success("Training and evaluation completed.")

            st.subheader("Workflow Summary")
            st.write("1. Train/test split with stratification")
            st.write("2. Outlier capping fitted on the training set and applied to the test set using the same bounds")
            st.write("3. RobustScaler fitted on the training set and applied to the test set using the same statistics")
            st.write("4. SMOTE applied within the training folds during cross-validation")
            st.write("5. Random Forest hyperparameter tuning on the training data with 5-fold cross-validation")
            st.write("6. Final retraining with the best hyperparameters and evaluation on the held-out test set")

            st.subheader("Key Metrics")
            metric_cols = st.columns(6)
            metric_items = [
                ("Accuracy", result["metrics"]["accuracy"]),
                ("Precision", result["metrics"]["precision"]),
                ("Recall", result["metrics"]["recall"]),
                ("F1 Score", result["metrics"]["f1_score"]),
                ("PR AUC", result["metrics"]["pr_auc"]),
            ]
            for col, (label, value) in zip(metric_cols, metric_items):
                col.metric(label, f"{value:.4f}")

            st.subheader("Best Hyperparameters")
            st.json(result["best_params"])

            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(result["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            st.subheader("Classification Report")
            report_df = pd.DataFrame(result["classification_report"]).T
            st.dataframe(report_df, use_container_width=True)

            st.subheader("Top Feature Importances")
            st.dataframe(result["feature_importance"], use_container_width=True)

            st.subheader("Insights")
            for insight in result["insights"]:
                st.write(f"- {insight}")

            st.subheader("Data Shape")
            st.write(f"Training rows: {result['train_shape'][0]} | Test rows: {result['test_shape'][0]}")

        except Exception as exc:
            st.error(f"Training failed: {exc}")
else:
    st.info("Please upload a CSV file to begin.")