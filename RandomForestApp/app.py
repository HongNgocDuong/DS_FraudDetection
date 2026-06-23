"""
Random Forest App - Standalone Application
For training and evaluating Random Forest models with dataset uploads
"""

import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page
st.set_page_config(
    page_title="🌲 Random Forest App",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding-top: 0rem;
    }
    </style>
    """, unsafe_allow_html=True)

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
ARTIFACT_DIR = APP_DIR / "artifacts"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)


# ==================== Helper Functions ====================

def load_data(file) -> pd.DataFrame | None:
    """Load CSV or Excel file into DataFrame."""
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file)
        else:
            st.error("❌ Please upload a CSV or Excel file.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None


def preprocess_data(df: pd.DataFrame, target_col: str, feature_cols: list) -> tuple:
    """Preprocess data for modeling."""
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    X = X.fillna(X.mean(numeric_only=True))
    y = y.fillna(y.mode()[0] if y.dtype == 'object' else y.mean())
    
    # Encode categorical features
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Encode target if classification
    target_encoder = None
    if y.dtype == 'object':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))
    
    return X, y, label_encoders, target_encoder


def save_model(model, metadata, filename="model"):
    """Save model and metadata."""
    model_path = ARTIFACT_DIR / f"{filename}.joblib"
    metadata_path = ARTIFACT_DIR / f"{filename}_metadata.json"
    
    joblib.dump(model, model_path)
    
    # Convert label encoders to serializable format
    serializable_metadata = {
        'feature_names': metadata.get('feature_names', []),
        'target_col': metadata.get('target_col', ''),
        'model_type': metadata.get('model_type', ''),
        'classes': model.classes_.tolist() if hasattr(model, 'classes_') else []
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(serializable_metadata, f, indent=2)
    
    return model_path, metadata_path


# ==================== Main App ====================

def main():
    # Header
    st.title("🌲 Random Forest Classifier & Regressor")
    st.markdown("**Build powerful predictive models with dataset uploads and interactive training**")
    st.divider()
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'preprocessor_info' not in st.session_state:
        st.session_state.preprocessor_info = None
    if 'test_results' not in st.session_state:
        st.session_state.test_results = None
    if 'training_data' not in st.session_state:
        st.session_state.training_data = None
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Training", "📈 Evaluation", "🔮 Prediction", "ℹ️ Guide"])
    
    # ==================== TAB 1: TRAINING ====================
    with tab1:
        st.header("Train Random Forest Model")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "📁 Upload your dataset",
                type=['csv', 'xlsx', 'xls'],
                help="CSV or Excel file with features and target variable"
            )
            
            if uploaded_file:
                df = load_data(uploaded_file)
                
                if df is not None:
                    st.session_state.training_data = df
                    st.success(f"✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
                    
                    with st.expander("📋 Data Preview"):
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        col_info1, col_info2, col_info3 = st.columns(3)
                        with col_info1:
                            st.metric("Rows", f"{df.shape[0]:,}")
                        with col_info2:
                            st.metric("Columns", df.shape[1])
                        with col_info3:
                            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                            st.metric("Missing %", f"{missing_pct:.1f}%")
                        
                        st.write("**Missing Values by Column:**")
                        st.bar_chart(df.isnull().sum())
                    
                    # Column selection
                    col_select1, col_select2 = st.columns(2)
                    
                    with col_select1:
                        target_col = st.selectbox(
                            "🎯 Select target variable",
                            options=df.columns,
                            help="The column you want to predict"
                        )
                    
                    with col_select2:
                        feature_cols = st.multiselect(
                            "🔧 Select features",
                            options=[col for col in df.columns if col != target_col],
                            default=[col for col in df.columns if col != target_col],
                            help="Columns to use for predictions"
                        )
                    
                    if not feature_cols:
                        st.warning("⚠️ Please select at least one feature.")
                    else:
                        # Model type selection
                        model_type = st.radio(
                            "Model Type",
                            options=["Classification", "Regression"],
                            horizontal=True,
                            help="Choose based on your target variable type"
                        )
                        
                        st.subheader("Model Hyperparameters")
                        
                        col_param1, col_param2, col_param3 = st.columns(3)
                        
                        with col_param1:
                            n_estimators = st.slider(
                                "Number of Trees",
                                min_value=10,
                                max_value=500,
                                value=100,
                                step=10
                            )
                        
                        with col_param2:
                            max_depth = st.slider(
                                "Max Depth",
                                min_value=1,
                                max_value=50,
                                value=15
                            )
                        
                        with col_param3:
                            min_samples_split = st.slider(
                                "Min Samples Split",
                                min_value=2,
                                max_value=20,
                                value=2
                            )
                        
                        col_param4, col_param5 = st.columns(2)
                        
                        with col_param4:
                            test_size = st.slider(
                                "Test Set Size (%)",
                                min_value=10,
                                max_value=50,
                                value=20,
                                step=5
                            ) / 100
                        
                        with col_param5:
                            random_state = st.number_input(
                                "Random State",
                                value=42,
                                help="For reproducibility"
                            )
                        
                        col_train1, col_train2 = st.columns([3, 1])
                        
                        with col_train1:
                            if st.button("🚀 Train Model", use_container_width=True, type="primary"):
                                with st.spinner("⏳ Training model..."):
                                    try:
                                        # Preprocess
                                        X, y, label_encoders, target_encoder = preprocess_data(
                                            df, target_col, feature_cols
                                        )
                                        
                                        # Split data
                                        X_train, X_test, y_train, y_test = train_test_split(
                                            X, y, test_size=test_size, random_state=int(random_state)
                                        )
                                        
                                        # Train model
                                        if model_type == "Classification":
                                            model = RandomForestClassifier(
                                                n_estimators=n_estimators,
                                                max_depth=max_depth,
                                                min_samples_split=min_samples_split,
                                                random_state=int(random_state),
                                                n_jobs=-1
                                            )
                                        else:
                                            model = RandomForestRegressor(
                                                n_estimators=n_estimators,
                                                max_depth=max_depth,
                                                min_samples_split=min_samples_split,
                                                random_state=int(random_state),
                                                n_jobs=-1
                                            )
                                        
                                        model.fit(X_train, y_train)
                                        
                                        # Make predictions
                                        y_pred_train = model.predict(X_train)
                                        y_pred_test = model.predict(X_test)
                                        
                                        # Store in session
                                        st.session_state.model = model
                                        st.session_state.preprocessor_info = {
                                            'feature_cols': feature_cols,
                                            'target_col': target_col,
                                            'label_encoders': label_encoders,
                                            'target_encoder': target_encoder,
                                            'model_type': model_type,
                                            'feature_names': feature_cols
                                        }
                                        st.session_state.test_results = {
                                            'X_test': X_test,
                                            'y_test': y_test,
                                            'y_pred_test': y_pred_test,
                                            'y_train': y_train,
                                            'y_pred_train': y_pred_train,
                                            'X_train': X_train
                                        }
                                        
                                        st.success("✅ Model trained successfully!")
                                        st.balloons()
                                        
                                        # Display metrics
                                        col_metric1, col_metric2 = st.columns(2)
                                        
                                        with col_metric1:
                                            st.subheader("📊 Training Metrics")
                                            if model_type == "Classification":
                                                train_acc = accuracy_score(y_train, y_pred_train)
                                                st.metric("Accuracy", f"{train_acc:.4f}", delta=None)
                                            else:
                                                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                                                st.metric("RMSE", f"{train_rmse:.4f}", delta=None)
                                        
                                        with col_metric2:
                                            st.subheader("📊 Testing Metrics")
                                            if model_type == "Classification":
                                                test_acc = accuracy_score(y_test, y_pred_test)
                                                st.metric("Accuracy", f"{test_acc:.4f}", delta=None)
                                            else:
                                                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                                                st.metric("RMSE", f"{test_rmse:.4f}", delta=None)
                                        
                                        st.divider()
                                        
                                        # Feature importance
                                        st.subheader("🎯 Feature Importance")
                                        feature_importance = pd.DataFrame({
                                            'Feature': feature_cols,
                                            'Importance': model.feature_importances_
                                        }).sort_values('Importance', ascending=False)
                                        
                                        fig, ax = plt.subplots(figsize=(12, 6))
                                        sns.barplot(data=feature_importance, x='Importance', y='Feature', 
                                                   palette='viridis', ax=ax)
                                        ax.set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
                                        ax.set_xlabel('Importance Score')
                                        st.pyplot(fig, use_container_width=True)
                                        
                                        st.divider()
                                        
                                        # Model download
                                        col_dl1, col_dl2 = st.columns(2)
                                        
                                        with col_dl1:
                                            # Download model
                                            model_bytes = io.BytesIO()
                                            joblib.dump(model, model_bytes)
                                            model_bytes.seek(0)
                                            
                                            st.download_button(
                                                label="📥 Download Model",
                                                data=model_bytes,
                                                file_name="random_forest_model.joblib",
                                                mime="application/octet-stream"
                                            )
                                        
                                        with col_dl2:
                                            # Download feature importance
                                            csv = feature_importance.to_csv(index=False)
                                            st.download_button(
                                                label="📥 Download Feature Importance",
                                                data=csv,
                                                file_name="feature_importance.csv",
                                                mime="text/csv"
                                            )
                                        
                                    except Exception as e:
                                        st.error(f"❌ Error during training: {e}")
        
        with col2:
            st.info(
                "**Quick Tips:**\n"
                "• Ensure clean data\n"
                "• More trees = better but slower\n"
                "• Adjust max depth for overfitting\n"
                "• Test size = % for validation"
            )
    
    # ==================== TAB 2: EVALUATION ====================
    with tab2:
        st.header("📊 Model Evaluation")
        
        if st.session_state.model is None:
            st.info("👈 Train a model first in the **Training** tab")
        else:
            results = st.session_state.test_results
            info = st.session_state.preprocessor_info
            
            if info['model_type'] == "Classification":
                st.subheader("Classification Evaluation")
                
                col_eval1, col_eval2 = st.columns(2)
                
                with col_eval1:
                    # Confusion Matrix
                    st.write("**Confusion Matrix**")
                    cm = confusion_matrix(results['y_test'], results['y_pred_test'])
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig, use_container_width=True)
                
                with col_eval2:
                    # Classification Metrics
                    st.write("**Performance Metrics**")
                    
                    accuracy = accuracy_score(results['y_test'], results['y_pred_test'])
                    precision = precision_score(results['y_test'], results['y_pred_test'], 
                                               average='weighted', zero_division=0)
                    recall = recall_score(results['y_test'], results['y_pred_test'], 
                                         average='weighted', zero_division=0)
                    f1 = f1_score(results['y_test'], results['y_pred_test'], 
                                 average='weighted', zero_division=0)
                    
                    met_col1, met_col2 = st.columns(2)
                    
                    with met_col1:
                        st.metric("Accuracy", f"{accuracy:.4f}")
                        st.metric("Recall", f"{recall:.4f}")
                    
                    with met_col2:
                        st.metric("Precision", f"{precision:.4f}")
                        st.metric("F1-Score", f"{f1:.4f}")
                
                # Classification Report
                st.subheader("📋 Detailed Classification Report")
                report = classification_report(
                    results['y_test'], 
                    results['y_pred_test'],
                    output_dict=True
                )
                
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
                
            else:
                st.subheader("Regression Evaluation")
                
                # Residuals plot
                residuals = results['y_test'] - results['y_pred_test']
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                # Residuals vs Predicted
                axes[0, 0].scatter(results['y_pred_test'], residuals, alpha=0.6, s=50)
                axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
                axes[0, 0].set_xlabel('Predicted Values')
                axes[0, 0].set_ylabel('Residuals')
                axes[0, 0].set_title('Residuals vs Predicted')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Histogram of residuals
                axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
                axes[0, 1].set_xlabel('Residuals')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Distribution of Residuals')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Actual vs Predicted
                min_val = min(results['y_test'].min(), results['y_pred_test'].min())
                max_val = max(results['y_test'].max(), results['y_pred_test'].max())
                axes[1, 0].scatter(results['y_test'], results['y_pred_test'], alpha=0.6, s=50)
                axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
                axes[1, 0].set_xlabel('Actual Values')
                axes[1, 0].set_ylabel('Predicted Values')
                axes[1, 0].set_title('Actual vs Predicted')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Q-Q plot
                from scipy import stats
                stats.probplot(residuals, dist="norm", plot=axes[1, 1])
                axes[1, 1].set_title('Q-Q Plot')
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                
                # Regression Metrics
                st.subheader("📊 Regression Metrics")
                
                rmse = np.sqrt(mean_squared_error(results['y_test'], results['y_pred_test']))
                mae = mean_absolute_error(results['y_test'], results['y_pred_test'])
                r2 = st.session_state.model.score(results['X_test'], results['y_test'])
                
                col_met1, col_met2, col_met3 = st.columns(3)
                
                with col_met1:
                    st.metric("RMSE", f"{rmse:.4f}")
                with col_met2:
                    st.metric("MAE", f"{mae:.4f}")
                with col_met3:
                    st.metric("R² Score", f"{r2:.4f}")
    
    # ==================== TAB 3: PREDICTION ====================
    with tab3:
        st.header("🔮 Make Predictions")
        
        if st.session_state.model is None:
            st.info("👈 Train a model first in the **Training** tab")
        else:
            info = st.session_state.preprocessor_info
            
            st.write(f"**Model Type:** {info['model_type']}")
            st.write(f"**Features:** {', '.join(info['feature_cols'])}")
            
            # Option to upload new data or enter manually
            pred_option = st.radio(
                "Prediction Input Method",
                ["Enter Manually", "Upload CSV"],
                horizontal=True
            )
            
            if pred_option == "Enter Manually":
                st.subheader("Enter Feature Values")
                
                input_data = {}
                cols = st.columns(2)
                
                for idx, feature in enumerate(info['feature_cols']):
                    with cols[idx % 2]:
                        input_data[feature] = st.number_input(
                            f"{feature}",
                            value=0.0,
                            format="%.2f"
                        )
                
                if st.button("🔮 Make Prediction", use_container_width=True, type="primary"):
                    try:
                        # Prepare data
                        X_input = pd.DataFrame([input_data])
                        
                        # Apply label encoders if needed
                        for col in info['feature_cols']:
                            if col in info['label_encoders']:
                                le = info['label_encoders'][col]
                                try:
                                    X_input[col] = le.transform([str(X_input[col].values[0])])[0]
                                except:
                                    st.warning(f"⚠️ Value not seen during training for {col}")
                        
                        prediction = st.session_state.model.predict(X_input)[0]
                        
                        # Decode prediction if classification
                        if info['model_type'] == "Classification" and info['target_encoder']:
                            prediction = info['target_encoder'].inverse_transform([int(prediction)])[0]
                        
                        st.success(f"✅ Prediction: **{prediction}**")
                        
                        # Show probabilities if classification
                        if info['model_type'] == "Classification":
                            st.subheader("Class Probabilities")
                            probabilities = st.session_state.model.predict_proba(X_input)[0]
                            
                            prob_data = []
                            for idx, prob in enumerate(probabilities):
                                class_label = st.session_state.model.classes_[idx]
                                if info['target_encoder']:
                                    try:
                                        class_label = info['target_encoder'].inverse_transform([class_label])[0]
                                    except:
                                        pass
                                prob_data.append({"Class": class_label, "Probability": f"{prob:.4f}"})
                            
                            prob_df = pd.DataFrame(prob_data)
                            st.dataframe(prob_df, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"❌ Prediction error: {e}")
            
            else:  # Upload CSV
                st.subheader("Batch Predictions")
                
                pred_file = st.file_uploader(
                    "📁 Upload dataset for predictions",
                    type=['csv', 'xlsx', 'xls']
                )
                
                if pred_file:
                    try:
                        pred_df = load_data(pred_file)
                        
                        if pred_df is not None:
                            # Check if all features are present
                            missing_features = set(info['feature_cols']) - set(pred_df.columns)
                            if missing_features:
                                st.error(f"❌ Missing features: {', '.join(missing_features)}")
                            else:
                                st.success(f"✅ All {len(info['feature_cols'])} required features found")
                                
                                X_pred = pred_df[info['feature_cols']].copy()
                                
                                # Apply label encoders
                                for col in info['feature_cols']:
                                    if col in info['label_encoders']:
                                        le = info['label_encoders'][col]
                                        try:
                                            X_pred[col] = le.transform(X_pred[col].astype(str))
                                        except:
                                            st.warning(f"⚠️ Some values in {col} not seen during training")
                                
                                if st.button("🔮 Make Batch Predictions", use_container_width=True, type="primary"):
                                    predictions = st.session_state.model.predict(X_pred)
                                    
                                    # Decode predictions if classification
                                    if info['model_type'] == "Classification" and info['target_encoder']:
                                        predictions = info['target_encoder'].inverse_transform(predictions)
                                    
                                    result_df = pred_df.copy()
                                    result_df['Prediction'] = predictions
                                    
                                    st.dataframe(result_df, use_container_width=True)
                                    
                                    # Download predictions
                                    csv = result_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Predictions",
                                        data=csv,
                                        file_name="predictions.csv",
                                        mime="text/csv"
                                    )
                    
                    except Exception as e:
                        st.error(f"❌ Error processing predictions: {e}")
    
    # ==================== TAB 4: GUIDE ====================
    with tab4:
        st.header("ℹ️ User Guide")
        
        st.subheader("📚 Getting Started")
        st.markdown("""
        ### Step 1: Prepare Your Data
        - Upload a CSV or Excel file
        - Ensure it contains both features and a target variable
        - The app automatically handles:
          - Missing values (filled with mean/mode)
          - Categorical variables (label encoded)
        
        ### Step 2: Train a Model
        1. Select your target variable (what to predict)
        2. Select features (what to use for prediction)
        3. Choose **Classification** or **Regression**
        4. Adjust hyperparameters if needed
        5. Click **Train Model**
        
        ### Step 3: Evaluate Results
        - View training and testing metrics
        - Analyze feature importance
        - For classification: confusion matrix and detailed metrics
        - For regression: residual plots and error metrics
        
        ### Step 4: Make Predictions
        - Enter values manually or upload a CSV
        - Get predictions and confidence scores
        - Download results
        """)
        
        st.divider()
        
        st.subheader("🤖 Model Types")
        
        col_guide1, col_guide2 = st.columns(2)
        
        with col_guide1:
            st.write("**Classification**")
            st.markdown("""
            - Predicts categories/classes
            - Example: Spam vs Not Spam
            - Output: Class labels + probabilities
            - Metrics: Accuracy, Precision, Recall, F1
            """)
        
        with col_guide2:
            st.write("**Regression**")
            st.markdown("""
            - Predicts continuous values
            - Example: House price prediction
            - Output: Numeric values
            - Metrics: RMSE, MAE, R²
            """)
        
        st.divider()
        
        st.subheader("⚙️ Hyperparameter Tuning")
        st.markdown("""
        | Parameter | Range | Effect |
        |-----------|-------|--------|
        | **Number of Trees** | 10-500 | More trees = better but slower |
        | **Max Depth** | 1-50 | Lower = less overfitting, simpler trees |
        | **Min Samples Split** | 2-20 | Higher = less overfitting |
        | **Test Size** | 10%-50% | Portion of data for validation |
        """)
        
        st.divider()
        
        st.subheader("💡 Tips & Tricks")
        st.markdown("""
        - **Data Quality**: Clean data produces better models
        - **Feature Selection**: Use relevant features only
        - **Imbalanced Data**: Add more minority class samples
        - **Overfitting**: Reduce max_depth or increase min_samples_split
        - **Underfitting**: Increase n_estimators or max_depth
        - **Feature Importance**: Check which features matter most
        """)
        
        st.divider()
        
        st.subheader("📊 Evaluation Metrics")
        
        with st.expander("Classification Metrics"):
            st.markdown("""
            - **Accuracy**: Overall correctness
            - **Precision**: True positives / all positive predictions
            - **Recall**: True positives / all actual positives
            - **F1-Score**: Harmonic mean of precision and recall
            """)
        
        with st.expander("Regression Metrics"):
            st.markdown("""
            - **RMSE**: Average magnitude of prediction errors
            - **MAE**: Mean absolute difference from predictions
            - **R²**: Proportion of variance explained (0-1)
            """)


if __name__ == "__main__":
    main()
