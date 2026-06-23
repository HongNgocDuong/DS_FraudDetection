# Random Forest App - Quick Start Guide

## Installation & Setup

### Step 1: Navigate to the Project
```powershell
cd C:\Users\AngieDuong\DS_FraudDetection\RandomForestApp
```

### Step 2: Create Virtual Environment (Optional)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run the App
```powershell
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## Project Structure

```
RandomForestApp/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # Full documentation
├── data/              # Upload your datasets here
└── artifacts/         # Trained models saved here
```

## Quick Features

✅ Upload CSV/Excel datasets  
✅ Train Classification & Regression models  
✅ Automatic data preprocessing  
✅ Interactive hyperparameter tuning  
✅ Feature importance analysis  
✅ Confusion matrices & residual plots  
✅ Manual & batch predictions  
✅ Download models & predictions  
✅ Built-in user guide  

## Common Tasks

### Upload and Train
1. Go to **Training** tab
2. Click "Upload your dataset"
3. Select target variable and features
4. Click "Train Model"

### Evaluate Model
1. Go to **Evaluation** tab
2. View confusion matrix (classification) or residual plots (regression)
3. Check metrics and feature importance

### Make Predictions
1. Go to **Prediction** tab
2. Choose manual entry or CSV upload
3. Enter values and click "Make Prediction"

## Support

See README.md for comprehensive documentation
