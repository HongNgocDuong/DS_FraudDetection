import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Upload & Split", layout="wide")
st.title("1. Upload Dataset and Create Train/Test Split")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state["df"] = df
    st.success("Dataset loaded.")
    st.dataframe(df.head(), use_container_width=True)

    target_options = [col for col in df.columns if col.lower() in {"class", "label", "target", "is_fraud", "fraud"}]
    if not target_options:
        target_options = df.columns.tolist()
    target_column = st.selectbox("Select the target column", target_options)
    st.session_state["target_column"] = target_column

    col1, col2 = st.columns(2)
    test_size = col1.slider("Test set size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    random_state = col2.number_input("Random state", min_value=1, max_value=1000, value=42)

    if st.button("Create train/test split"):
        X = df.drop(columns=[target_column]).copy()
        y = df[target_column].astype(str)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=int(random_state)
        )
        st.session_state["X_train"] = X_train.reset_index(drop=True)
        st.session_state["X_test"] = X_test.reset_index(drop=True)
        st.session_state["y_train"] = y_train.reset_index(drop=True)
        st.session_state["y_test"] = y_test.reset_index(drop=True)
        st.session_state["test_size"] = test_size
        st.session_state["random_state"] = int(random_state)
        st.success("Train/test split created.")
        st.write(f"Training rows: {len(X_train)}")
        st.write(f"Test rows: {len(X_test)}")
else:
    st.info("Upload a CSV file to begin.")
