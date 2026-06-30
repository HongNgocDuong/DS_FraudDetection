import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("Fraud Detection App")
st.write(
    "This workflow follows the requested sequence: split into train/test, preprocess the training data with outlier capping and RobustScaler, tune XGBoost, retrain with the best hyperparameters, and evaluate on the test set."
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


def cap_outliers_3sigma(train_df, test_df):
    train_capped = train_df.copy()
    test_capped = test_df.copy()
    for col in train_df.columns:
        if not pd.api.types.is_numeric_dtype(train_df[col]):
            continue
        if train_df[col].nunique(dropna=True) <= 1:
            continue
        mean = train_df[col].mean()
        std = train_df[col].std()
        if pd.isna(std) or std == 0:
            continue
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        train_capped[col] = train_capped[col].clip(lower=lower_bound, upper=upper_bound)
        test_capped[col] = test_capped[col].clip(lower=lower_bound, upper=upper_bound)
    return train_capped, test_capped


def build_preprocessor(X_train, X_test):
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError("The uploaded data does not contain any usable feature columns.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    X_train_prepared = preprocessor.fit_transform(X_train)
    X_test_prepared = preprocessor.transform(X_test)
    return preprocessor, X_train_prepared, X_test_prepared


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

    X_train_capped, X_test_capped = cap_outliers_3sigma(X_train, X_test)
    preprocessor, X_train_prepared, X_test_prepared = build_preprocessor(X_train_capped, X_test_capped)

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    tuner = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=12,
        scoring="average_precision",
        cv=3,
        n_jobs=-1,
        random_state=random_state,
        verbose=0,
    )
    tuner.fit(X_train_prepared, y_train)

    best_params = tuner.best_params_
    final_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        **best_params,
    )
    final_model.fit(X_train_prepared, y_train)
    predictions = final_model.predict(X_test_prepared)
    probabilities = final_model.predict_proba(X_test_prepared)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_test, probabilities)), 4),
    }

    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=["Negative", "Positive"], output_dict=True)

    feature_names = preprocessor.get_feature_names_out()
    importances = pd.DataFrame(
        {"feature": feature_names, "importance": final_model.feature_importances_}
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

    if metrics["roc_auc"] > 0.85:
        insights.append("The ROC AUC is strong, suggesting the ranking quality is good for fraud detection.")
    else:
        insights.append("The ROC AUC is moderate, so the model could benefit from more tuning or additional engineered features.")

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
            st.write("2. Outlier capping fitted on the training set and applied to both train/test")
            st.write("3. RobustScaler and imputation applied after fitting on the training set")
            st.write("4. XGBoost hyperparameter tuning on the training data")
            st.write("5. Final retraining with the best hyperparameters and evaluation on the held-out test set")

            st.subheader("Key Metrics")
            metric_cols = st.columns(6)
            metric_items = [
                ("Accuracy", result["metrics"]["accuracy"]),
                ("Precision", result["metrics"]["precision"]),
                ("Recall", result["metrics"]["recall"]),
                ("F1 Score", result["metrics"]["f1_score"]),
                ("ROC AUC", result["metrics"]["roc_auc"]),
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