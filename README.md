# Telecom Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn, using the IBM Telco Customer Churn dataset. The goal is to help businesses identify high-risk customers early and take targeted retention action before losing them.

---

## Problem Statement

Customer churn is one of the most costly problems in the telecom industry. Acquiring a new customer costs 5–7x more than retaining an existing one. This project builds a binary classification model that predicts churn likelihood from customer demographics, account information, and service usage — enabling proactive retention campaigns.

---

##  Dataset

- **Source:** [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers × 21 features
- **Target variable:** `Churn` (Yes / No)
- **Key features:** tenure, contract type, monthly charges, internet service, payment method

> Download the dataset from the link above and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project root directory before running the notebook.

---

##  Project Structure

```
telecom-churn-prediction/
│
├── Customer_Churn_Prediction_using_ML.ipynb   # Main notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv       # Dataset (download separately)
├── customer_churn_model.pkl                   # Saved trained model
└── README.md
```

---

## Methodology

### 1. Data Cleaning
- Removed `customerID` column (non-predictive identifier)
- Fixed 11 rows where `TotalCharges` was stored as whitespace `" "` — replaced with `0.0` and cast to float
- Confirmed zero null values across all 20 remaining features

### 2. Exploratory Data Analysis (EDA)
- Distribution analysis of numerical features: `tenure`, `MonthlyCharges`, `TotalCharges`
- Box plots to detect outliers
- Correlation heatmap across numerical features
- Count plots for all 17 categorical features vs churn rate
- Key finding: customers on month-to-month contracts with high monthly charges and short tenure are significantly more likely to churn

### 3. Data Preprocessing
- Label encoded all categorical features using `sklearn.LabelEncoder`
- Applied **SMOTE** (Synthetic Minority Oversampling Technique) to address class imbalance (73% No / 27% Yes → balanced to 50/50 on training data)

### 4. Model Training & Selection
Trained three models with 5-fold cross-validation on SMOTE-balanced training data:

| Model | CV Accuracy |
|---|---|
| Decision Tree | 0.78 |
| Random Forest | **0.84** |
| XGBoost | 0.82 |

**Random Forest selected** as best-performing model.

### 5. Model Evaluation (Test Set)

| Metric | Score |
|---|---|
| Accuracy | 81% |
| AUC-ROC | 0.86 |
| Precision (Churn) | 0.85 |
| F1-Score (Churn) | 0.85 |

### 6. Predictive System
- Trained model serialised with `pickle`
- End-to-end inference function accepts raw customer input, applies the same label encoding pipeline, and returns churn prediction with probability score

---

## Key Findings

- **Contract type** is the strongest churn predictor — month-to-month customers churn at 3x the rate of two-year contract customers
- **Tenure** has a strong inverse relationship with churn — customers who stay beyond 12 months show significantly lower churn risk
- **Monthly charges above ₹65** correlate with higher churn, especially without bundled services
- SMOTE meaningfully improved recall on the minority churn class without significantly reducing precision

---

##  Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data manipulation | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| ML models | scikit-learn, xgboost |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Model persistence | pickle |
| Environment | Jupyter Notebook / Google Colab |

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/kanishkdutta22/telecom-churn-prediction
cd telecom-churn-prediction
```

2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the project root

4. Open and run the notebook
```bash
jupyter notebook Customer_Churn_Prediction_using_ML.ipynb
```

> Run all cells top to bottom. The notebook is fully reproducible with `random_state=42` set throughout.

---

## Future Improvements

- Hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`
- Feature importance analysis and feature selection
- Try ensemble stacking with XGBoost + Random Forest
- Build a simple Streamlit web app for live churn prediction
- Experiment with downsampling and compare against SMOTE results

---

##  Author

**Kanishk Dutta**  
MSc Economics — Gokhale Institute of Politics and Economics  
[LinkedIn](https://linkedin.com/in/kanishkdutta22) · [GitHub](https://github.com/kanishkdutta22) · kanishkdutta22@gmail.com
