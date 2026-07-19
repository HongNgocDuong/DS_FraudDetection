from pathlib import Path

import pandas as pd
import streamlit as st


def get_sample_dataset_path() -> Path:
    return Path(__file__).resolve().parent / "Presentation deck & Python notebook" / "fraud_sample_dataset.csv"


def load_demo_dataframe(dataset_path: Path | None = None) -> tuple[pd.DataFrame, str, str]:
    candidate_path = dataset_path or get_sample_dataset_path()

    if candidate_path.exists():
        try:
            df = pd.read_csv(candidate_path, skip_blank_lines=True, encoding="utf-8-sig")
            if not df.empty and df.shape[1] > 0:
                return df, str(candidate_path), candidate_path.name
        except Exception:
            pass

    fallback_df = pd.DataFrame(
        {
            "Time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "Amount": [100.0, 120.5, 80.0, 500.0, 75.0, 60.0, 200.0, 90.0, 300.0, 110.0, 150.0, 450.0],
            "V1": [-1.2, -0.8, 0.4, 2.3, -0.3, 0.5, -1.1, 0.2, 1.0, -0.6, 0.9, 2.1],
            "V2": [0.2, -0.1, 0.5, 1.2, -0.9, 0.3, -0.8, 0.7, 0.4, -0.5, 0.6, 1.1],
            "Class": [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
        }
    )
    return fallback_df, "fallback", "fallback_demo_dataset.csv"


def ensure_static_demo_state() -> bool:
    if st.session_state.get("static_demo_ready"):
        return True

    df, dataset_source, dataset_name = load_demo_dataframe()

    st.session_state["dataset_source"] = dataset_source
    st.session_state["dataset_name"] = dataset_name

    target_column = "Class" if "Class" in df.columns else "label" if "label" in df.columns else df.columns[-1]
    feature_columns = [col for col in df.columns if col != target_column]
    numeric_columns = [col for col in feature_columns if pd.api.types.is_numeric_dtype(df[col])]

    st.session_state["df"] = df
    st.session_state["target_column"] = target_column
    st.session_state["feature_columns"] = feature_columns
    st.session_state["numeric_columns"] = numeric_columns
    st.session_state["dataset_shape"] = df.shape

    st.session_state["split_summary"] = {
        "train_rows": int(len(df) * 0.8),
        "test_rows": int(len(df) * 0.2),
        "target_column": target_column,
        "test_size": 0.2,
    }

    st.session_state["preprocess_summary"] = {
        "folds": 5,
        "scaled_columns": [col for col in ["Time", "Amount"] if col in numeric_columns],
        "outlier_capping": True,
        "smote": True,
        "feature_count": len(feature_columns),
    }

    st.session_state["model_summary"] = {
        "model": "Random Forest",
        "best_params": {
            "n_estimators": 200,
            "max_depth": 7,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
        },
        "cv_score": 0.91,
    }

    st.session_state["evaluation_summary"] = {
        "accuracy": 0.9698,
        "precision": 0.9524,
        "recall": 0.9231,
        "f1_score": 0.9375,
        "pr_auc": 0.9312,
        "confusion_matrix": [[182, 6], [3, 109]],
    }

    st.session_state["step_complete_1"] = True
    st.session_state["step_complete_2"] = True
    st.session_state["step_complete_3"] = True
    st.session_state["step_complete_4"] = True
    st.session_state["latest_completed_step"] = 4
    st.session_state["current_step"] = 5
    st.session_state["static_demo_ready"] = True
    st.session_state["demo_mode"] = "static"
    return True
