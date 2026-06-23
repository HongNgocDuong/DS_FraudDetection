# 🌲 Random Forest App

A standalone Streamlit application for training, evaluating, and making predictions with Random Forest models.

## Features

### 📊 Training Tab
- Upload CSV or Excel datasets
- Interactive data exploration and statistics
- Feature and target variable selection
- Classification and Regression model support
- Tunable hyperparameters:
  - Number of trees (10-500)
  - Max depth (1-50)
  - Minimum samples split (2-20)
  - Train/test split ratio (10%-50%)
- Feature importance visualization
- Model download functionality

### 📈 Evaluation Tab
**For Classification:**
- Confusion matrix heatmap
- Classification metrics (Accuracy, Precision, Recall, F1-Score)
- Detailed classification report

**For Regression:**
- Residuals analysis plots
- Actual vs. Predicted scatter plot
- Distribution analysis and Q-Q plot
- Regression metrics (RMSE, MAE, R² Score)

### 🔮 Prediction Tab
- Manual single predictions
- Batch predictions via CSV upload
- Class probability estimates
- Results download as CSV

### ℹ️ Guide Tab
- Comprehensive user documentation
- Model type explanations
- Hyperparameter tuning guide
- Tips and best practices
- Metric explanations

## Installation

### 1. Clone or Download the Repository
```bash
git clone <repo-url>
cd RandomForestApp
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Start the App
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Workflow

1. **Prepare Data**
   - Upload a CSV or Excel file with features and target variable
   - View data statistics and missing value analysis

2. **Select Variables**
   - Choose which column to predict (target)
   - Select columns to use as features

3. **Configure Model**
   - Choose Classification or Regression
   - Adjust hyperparameters as needed

4. **Train**
   - Click "Train Model"
   - View training and testing performance

5. **Evaluate**
   - Check confusion matrix (classification) or residual plots (regression)
   - Analyze feature importance
   - Review detailed metrics

6. **Predict**
   - Enter values manually or upload new data
   - Get predictions and download results

7. **Export**
   - Download trained models
   - Save feature importance rankings

## Example Datasets

### Classification Example
```
age,income,credit_score,loan_approved
25,35000,650,0
45,95000,750,1
32,55000,700,1
```

### Regression Example
```
bedrooms,sqft,age,price
3,2000,10,350000
4,3000,5,550000
2,1500,20,250000
```

## Model Characteristics

### Random Forest Classifier
- Multi-class and binary classification
- Handles imbalanced datasets
- Captures non-linear relationships
- Built-in feature importance
- Returns class probabilities

### Random Forest Regressor
- Continuous value prediction
- Robust to outliers
- Non-linear relationship modeling
- Feature importance analysis
- Good generalization

## Hyperparameter Guide

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| n_estimators | 100 | 10-500 | More trees = better but slower |
| max_depth | 15 | 1-50 | Lower = simpler, less overfitting |
| min_samples_split | 2 | 2-20 | Higher = less complex trees |
| test_size | 0.2 | 0.1-0.5 | Validation data proportion |

## Metrics Explained

### Classification
- **Accuracy**: Percentage of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of Precision and Recall

### Regression
- **RMSE**: Root Mean Squared Error - penalizes large errors more
- **MAE**: Mean Absolute Error - average magnitude of errors
- **R²**: Coefficient of determination - variance explained (0-1 scale)

## Project Structure

```
RandomForestApp/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── data/              # Sample datasets directory
├── artifacts/         # Saved models directory
└── utils/             # Utility functions (optional)
```

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: "File not found" error
**Solution**: Ensure your CSV/Excel file is properly formatted and located correctly

### Issue: Low model accuracy
**Solutions**:
- Use more relevant features
- Ensure sufficient training data (recommend 100+ rows)
- Check for data quality issues
- Adjust hyperparameters

### Issue: Out of memory
**Solutions**:
- Reduce n_estimators (number of trees)
- Reduce max_depth
- Use a smaller dataset

## Tips for Best Results

1. **Data Quality**
   - Remove duplicate rows
   - Handle missing values appropriately
   - Ensure consistent data types

2. **Feature Engineering**
   - Select relevant features only
   - Remove highly correlated features
   - Scale numeric features if needed

3. **Hyperparameter Tuning**
   - Start with defaults
   - Adjust based on results
   - Watch for overfitting (training >> testing accuracy)

4. **Validation**
   - Use appropriate test_size (20% is common)
   - Check feature importance
   - Review error patterns

## Supported File Formats

- CSV (.csv)
- Excel (.xlsx, .xls)

## Dependencies

- **streamlit**: Web app framework
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Visualization
- **scipy**: Statistical functions
- **joblib**: Model serialization

## Performance Considerations

- **Dataset Size**: 100-10,000 rows ideal
- **Number of Features**: 5-50 features recommended
- **Training Time**: 
  - Small datasets: <1 second
  - Medium datasets: 1-10 seconds
  - Large datasets: 10+ seconds

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues and enhancement requests!

## Support

For issues, questions, or suggestions, please open an issue in the repository.

---

**Version**: 1.0  
**Last Updated**: 2026-06-23  
**Author**: Data Analytics Team
