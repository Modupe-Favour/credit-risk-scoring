# 🏦 End-to-End Credit Risk Scoring Model

A production-ready machine learning pipeline that predicts loan
default risk and explains model decisions using SHAP values.

## 🔗 Live Demo
[Click here to view the live dashboard](https://accessibilityquiz-hukqgornpgtwaqnxzv6mew.streamlit.app/)

## 📌 Project Overview
Credit risk assessment is one of the most critical functions in
financial institutions. This project builds an end-to-end ML
pipeline that:
- Analyses default patterns across loan applicant data
- Engineers meaningful financial features
- Trains and compares Logistic Regression, XGBoost, and LightGBM
- Handles class imbalance using SMOTE
- Deploys an interactive dashboard with live predictions
- Explains model decisions using SHAP values

## 📊 Key Findings
- Default rate is ~8% — dataset is significantly imbalanced
- Younger applicants (20–30) carry the highest default risk
- Credit-to-income ratio is the strongest predictor of default
- Employment stability strongly influences risk level

## 🛠️ Tools & Technologies
| Area | Tools |
|---|---|
| Language | Python 3.11 |
| Data Analysis | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Dashboard | Streamlit |
| IDE | VS Code |

## 🚀 How to Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-scoring.git
cd credit-risk-scoring
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## 📁 Project Structure
```
credit-risk-scoring/
├── Data/                   # Dataset
├── Notebook/               # EDA notebook
├── app.py                  # Streamlit dashboard
├── eda.py                  # EDA functions
├── train_model.py          # Model training script
├── model.pkl               # Saved model
├── requirements.txt
└── README.md
```

## 💼 Business Impact
This tool enables loan officers to make faster, data-driven
decisions while providing transparent explanations for every
prediction — critical for regulatory compliance in finance.
