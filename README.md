Real-Time UPI Transaction Fraud Detection System

Overview

This project implements a Machine Learning-based fraud detection system for UPI (Unified Payments Interface) transactions. The system analyzes transaction patterns in real-time to identify and prevent fraudulent activities, ensuring secure digital payments.

Key Features

- **Real-time Transaction Monitoring**: Analyzes UPI transactions as they occur
- **Advanced ML Models**: Random Forest and Logistic Regression with optimized hyperparameters
- **Feature Engineering**: Comprehensive feature extraction from transaction data
- **Class Imbalance Handling**: SMOTE technique for effective fraud detection
- **High Performance**: 99.86% accuracy with Random Forest model
- **Visual Analytics**: Interactive plots and performance metrics

 Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| **Random Forest** | **99.86%** | **99.89%** | **99.82%** | **99.86%** | **99.97%** |
| Logistic Regression | 55.77% | 55.28% | 60.40% | 57.73% | 56.74% |

 Dataset Features

| Category | Features |
|----------|----------|
| **Transaction Info** | Transaction ID, Type, Amount, Status |
| **User Details** | Sender/Receiver Age Groups, States, Banks |
| **Technical** | Device Type, Network Type, Timestamp |
| **Temporal** | Hour, Day, Weekend Flags |
| **Target** | Fraud Flag (Binary) |

Technical Architecture

### Data Pipeline
```
Raw UPI Data → Feature Engineering → Data Preprocessing → Model Training → Real-time Prediction
```

### Feature Engineering
- **Temporal Features**: Month, day, hour, day-of-week encoding
- **Time-based Categories**: Morning, afternoon, evening, night flags
- **Amount Features**: Log transformation, amount categorization
- **Transaction Patterns**: Same sender-receiver bank detection
- **Categorical Encoding**: Label encoding for all categorical variables

### Model Selection
- **Random Forest**: Primary model with ensemble learning
- **Logistic Regression**: Baseline model for comparison
- **SMOTE**: Synthetic Minority Over-sampling Technique for class balance
Results & Visualizations

The system generates comprehensive visualizations:

- **Confusion Matrix**: Model prediction accuracy
- **ROC Curve**: Model discrimination ability
- **Precision-Recall Curve**: Fraud detection performance
- **Feature Importance**: Key factors in fraud detection

 **Model Configuration**

**Random Forest Parameters**

RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced'
)

**Logistic Regression Parameters**

LogisticRegression(
    random_state=42,
    class_weight='balanced',
    max_iter=1000
)


 **Business Impact**

For Payment Companies (like PhonePe)
- **Fraud Prevention**: Reduces financial losses from fraudulent transactions
- **User Trust**: Maintains customer confidence in the platform
- **Regulatory Compliance**: Meets financial security requirements
- **Cost Savings**: Automated detection reduces manual review costs

 For Users
- **Enhanced Security**: Real-time protection against fraud
- **Seamless Experience**: Minimal false positives
- **Peace of Mind**: Secure digital transactions

