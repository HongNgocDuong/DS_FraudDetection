import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("Fraud Detection App")
st.write(
    "Upload a CSV file, choose the target column, and train an XGBoost classifier to detect fraud-like outcomes."
)


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)


def cap_outliers_3sigma(df, numeric_columns):
    df_capped = df.copy()
    for col in numeric_columns:
        if col not in df_capped.columns:
            continue
        mean = df_capped[col].mean()
        std = df_capped[col].std()
        if pd.isna(std) or std == 0:
            continue
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
    return df_capped


def encode_target(series):
    clean = series.astype(str).str.strip()
    unique_values = sorted(clean.dropna().unique())
    if len(unique_values) != 2:
        raise ValueError("The target column must contain exactly two classes for binary classification.")
    mapping = {unique_values[0]: 0, unique_values[1]: 1}
    return clean.map(mapping).astype(int)


def build_preprocessor(X):
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()

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
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError("The uploaded data does not contain any usable feature columns.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_and_train(df, target_column, test_size, use_smote, random_state=42):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the uploaded data.")

    X = df.drop(columns=[target_column]).copy()
    y = encode_target(df[target_column])

    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        X_train_capped = cap_outliers_3sigma(X_train, numeric_columns)
        X_test_capped = cap_outliers_3sigma(X_test, numeric_columns)
        X_train_capped = X_train_capped.reset_index(drop=True)
        X_test_capped = X_test_capped.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        X_train_capped = X_train.reset_index(drop=True)
        X_test_capped = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

    preprocessor = build_preprocessor(X_train_capped)
    xgb_model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
    )

    if use_smote:
        model = ImbPipeline(
            steps=[
                ("preprocess", preprocessor),
                ("smote", SMOTE(random_state=random_state)),
                ("classifier", xgb_model),
            ]
        )
    else:
        model = Pipeline(steps=[("preprocess", preprocessor), ("classifier", xgb_model)])

    model.fit(X_train_capped, y_train)
    predictions = model.predict(X_test_capped)
    probabilities = model.predict_proba(X_test_capped)[:, 1]

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

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.named_steps["classifier"].feature_importances_,
        }
    ).sort_values(by="importance", ascending=False).head(15)

    return {
        "model": model,
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "feature_importance": importances,
        "y_test": y_test,
        "predictions": predictions,
        "probabilities": probabilities,
        "train_shape": X_train_capped.shape,
        "test_shape": X_test_capped.shape,
    }


uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    with st.spinner("Loading data..."):
        df = load_csv(uploaded_file)

    st.subheader("Preview")
    st.dataframe(df.head(), use_container_width=True)

    target_options = [col for col in df.columns if col.lower() in {"class", "label", "target", "is_fraud", "fraud"}]
    if not target_options:
        target_options = df.columns.tolist()
    target_column = st.selectbox("Select the target column", target_options)

    col1, col2, col3 = st.columns(3)
    test_size = col1.slider("Test set size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    use_smote = col2.checkbox("Apply SMOTE to balance the minority class", value=True)
    random_state = col3.number_input("Random state", min_value=1, max_value=1000, value=42)

    if st.button("Train XGBoost model"):
        try:
            with st.spinner("Training the model..."):
                result = prepare_and_train(
                    df=df,
                    target_column=target_column,
                    test_size=test_size,
                    use_smote=use_smote,
                    random_state=int(random_state),
                )

            st.success("Model training completed.")
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

            st.subheader("Preprocessing Summary")
            st.write(
                f"Training rows: {result['train_shape'][0]} | Test rows: {result['test_shape'][0]}"
            )
            st.write("Applied preprocessing steps: outlier capping, median/most-frequent imputation, robust scaling, one-hot encoding, and optional SMOTE.")

        except Exception as exc:
            st.error(f"Training failed: {exc}")
else:
    st.info("Please upload a CSV file to begin.")