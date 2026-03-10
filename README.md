# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using logistic regression with class imbalance handling.

## Project Overview

This project implements a credit card fraud detection system using machine learning techniques. The system is designed to identify fraudulent transactions in a highly imbalanced dataset where fraudulent transactions represent only 0.17% of all transactions.

## Dataset

The project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle, which contains:
- **284,807 transactions** over a 2-day period
- **492 fraudulent transactions** (0.172% of total)
- **31 features**: 28 anonymized V1-V28 features, Time, and Amount
- **Class label**: 0 for legitimate, 1 for fraudulent

### Features
- **Time**: Seconds elapsed between transaction and first transaction
- **V1-V28**: Anonymized features from PCA transformation
- **Amount**: Transaction amount
- **Class**: Target variable (0 = legitimate, 1 = fraudulent)

## Project Structure

```
credit_card_fraud_detection/
├── dataset/
│   └── creditcard.csv          # Raw dataset
├── notebooks/
│   ├── eda.ipynb              # Exploratory Data Analysis
│   └── training.ipynb         # Model Training and Evaluation
├── models/
│   └── linear_regression/     # Trained models and scalers
├── pyproject.toml             # Project dependencies
├── .python-version            # Python version

```

## Exploratory Data Analysis (EDA)

The EDA notebook (`notebooks/eda.ipynb`) provides comprehensive analysis including:

### Data Overview
- Dataset shape: 284,807 rows × 31 columns
- No missing values in any feature
- All features are numeric (float64)

### Class Imbalance Analysis
- **Legitimate transactions**: 284,315 (99.83%)
- **Fraudulent transactions**: 492 (0.17%)
- Severe class imbalance requiring specialized handling

![Class Distribution](public/images/eda/dataset-vis.png)

### Feature Analysis
- **Transaction Amount**: 
  - Range: $0 - $25,691
  - Most transactions between $100-$1,000
  - Fraudulent transactions tend to have higher amounts

![Transaction Amount Distribution](public/images/eda/transaction-amount-vs-count.png)

![Transaction Amount Distribution (Log Scale)](public/images/eda/transaction-amount-vs-count-logscale.png)

- **Time Analysis**:
  - Transactions distributed across 24-hour periods
  - Fraudulent transactions show different temporal patterns
  - Peak fraud activity observed during specific hours

![Legitimate Transactions by Hour](public/images/eda/legit-transaction-count-by-hour.png)

![Fraudulent Transactions by Hour](public/images/eda/fraud-transaction-count-by-hour.png)

- **Feature Distributions**:
  - V1-V28 features show different distributions for fraud vs legitimate
  - Box plots reveal outliers and distribution differences
  - Correlation heatmap shows feature relationships

## Model Training

The training notebook (`notebooks/training.ipynb`) implements:

### Data Preprocessing
- **Train/Test Split**: 80/20 stratified split
- **Feature Scaling**: StandardScaler applied to all features
- **Class Weighting**: Balanced class weights to handle imbalance

### Model Architecture
- **Algorithm**: Logistic Regression
- **Class Weight**: Balanced (automatically adjusts weights inversely proportional to class frequencies)
- **Max Iterations**: 5,000 (to ensure convergence)
- **Regularization**: L2 (default)

### Performance Metrics
Due to class imbalance, we focus on:
- **Precision**: Proportion of predicted frauds that are actually fraudulent
- **Recall**: Proportion of actual frauds correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the Receiver Operating Characteristic curve

### Results
- **Precision**: High precision ensures low false positive rate
- **Recall**: Good recall captures most fraudulent transactions
- **F1-Score**: Balanced performance between precision and recall
- **ROC AUC**: Excellent discrimination capability

## Installation

### Prerequisites
- Python 3.12+
- pip or uv package manager

### Using uv (Recommended)
```bash
# Install dependencies
uv add pandas scikit-learn matplotlib numpy mlflow joblib

# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### Using pip
```bash
# Install dependencies
pip install pandas scikit-learn matplotlib numpy mlflow joblib

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

## Usage

### 1. Data Exploration
```bash
# Run EDA notebook
jupyter notebook notebooks/eda.ipynb
```

### 2. Model Training
```bash
# Run training notebook
jupyter notebook notebooks/training.ipynb
```

### 3. Model Inference
The trained model can be used for fraud detection:

```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('models/linear_regression/linear_reg_baseline_model.pkl')
scaler = joblib.load('models/linear_regression/linear_reg_scaler.pkl')

# Prepare transaction data
transaction = {
    'Time': 10000,
    'V1': -1.359807,
    'V2': -0.072781,
    # ... other features
    'Amount': 149.62
}

# Make prediction
transaction_df = pd.DataFrame([transaction])
transaction_scaled = scaler.transform(transaction_df)
prediction = model.predict(transaction_scaled)
probability = model.predict_proba(transaction_scaled)[0][1]

if prediction[0] == 1:
    print(f"Fraud detected! Probability: {probability:.4f}")
else:
    print(f"Legitimate transaction. Fraud probability: {probability:.4f}")
```

## Model Monitoring with MLflow

The project uses MLflow for experiment tracking:

### Starting MLflow Server
```bash
mlflow server --host 127.0.0.1 --port 5000
```

### Viewing Experiments
Access the MLflow UI at `http://127.0.0.1:5000` to view:
- Experiment parameters
- Model metrics (precision, recall, F1-score)
- Model artifacts
- ROC curves and performance visualizations

![MLflow Dashboard](public/images/mlflow/mlflow-dashboard.png)

![MLflow Metrics](public/images/mlflow/mlflow-metrics.png)

## Key Findings

### Data Characteristics
1. **Severe Class Imbalance**: Only 0.17% of transactions are fraudulent
2. **Feature Scaling Required**: Amount feature has different scale than V1-V28
3. **Temporal Patterns**: Fraudulent transactions show different time distributions



### Feature Importance
- V1-V28 features contain valuable information for fraud detection
- Amount and Time features provide additional discriminatory power
- Standard scaling improves model convergence and performance


