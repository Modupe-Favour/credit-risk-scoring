import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from eda import (load_data, default_distribution, default_by_intent,
                 default_by_age, default_by_loan_amount,
                 correlation_heatmap)

# --- Page Config ---
st.set_page_config(page_title="Credit Risk Scoring",
                   layout="wide", page_icon="🏦")

# --- Load Assets ---
df = load_data("Data/credit_risk.csv")
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# --- Sidebar ---
st.sidebar.title("🏦 Credit Risk Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", [
    "📊 EDA & Insights",
    "🔮 Predict Default Risk",
    "🧠 SHAP Explainability",
    "📋 Business Recommendations"
])

# ============================================================
# PAGE 1 — EDA & INSIGHTS
# ============================================================
if page == "📊 EDA & Insights":
    st.title("📊 Credit Risk Analysis")
    st.markdown("Explore default patterns across loan applicants.")
    st.markdown("---")

    total = len(df)
    defaults = int(df["loan_status"].sum())
    no_defaults = total - defaults
    default_rate = round(df["loan_status"].mean() * 100, 2)
    avg_loan = round(df["loan_amnt"].mean(), 2)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Applicants", f"{total:,}")
    c2.metric("Defaults", f"{defaults:,}")
    c3.metric("No Defaults", f"{no_defaults:,}")
    c4.metric("Default Rate", f"{default_rate}%")
    c5.metric("Avg Loan Amount", f"${avg_loan:,.0f}")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(default_distribution(df), use_container_width=True)
    with col2:
        st.plotly_chart(default_by_intent(df), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(default_by_age(df), use_container_width=True)
    with col4:
        st.plotly_chart(default_by_loan_amount(df),
                        use_container_width=True)

    st.subheader("🔥 Correlation Heatmap")
    st.pyplot(correlation_heatmap(df))

    st.markdown("---")
    st.subheader("🤖 Model Performance")
    col5, col6 = st.columns(2)
    with col5:
        st.image("roc_curve.png", caption="ROC Curve")
    with col6:
        st.image("confusion_matrix.png", caption="Confusion Matrix")
    if st.checkbox("Show Feature Importance"):
        st.image("feature_importance.png", caption="Top 10 Features")

# ============================================================
# PAGE 2 — PREDICT DEFAULT RISK
# ============================================================
elif page == "🔮 Predict Default Risk":
    st.title("🔮 Predict Loan Default Risk")
    st.markdown("Enter applicant details to predict default probability.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Personal Info")
        person_age = st.slider("Age", 18, 70, 30)
        person_income = st.number_input("Annual Income ($)",
                                        0, 1000000, 50000)
        person_emp_length = st.slider("Employment Length (years)",
                                       0, 40, 5)
        person_home_ownership = st.selectbox("Home Ownership",
                                              ["RENT", "OWN",
                                               "MORTGAGE", "OTHER"])

    with col2:
        st.subheader("Loan Details")
        loan_amnt = st.number_input("Loan Amount ($)", 0, 100000, 10000)
        loan_int_rate = st.number_input("Interest Rate (%)",
                                         0.0, 30.0, 12.0)
        loan_intent = st.selectbox("Loan Purpose",
                                    ["PERSONAL", "EDUCATION",
                                     "MEDICAL", "VENTURE",
                                     "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        loan_grade = st.selectbox("Loan Grade",
                                   ["A", "B", "C", "D", "E", "F", "G"])

    with col3:
        st.subheader("Credit Info")
        loan_percent_income = round(loan_amnt / (person_income + 1), 2)
        st.metric("Loan % of Income", f"{loan_percent_income * 100:.1f}%")
        cb_default = st.selectbox("Previous Default on File",
                                   ["N", "Y"])
        cb_cred_hist = st.slider("Credit History Length (years)",
                                  0, 30, 5)

    # --- Build Input ---
    le = LabelEncoder()

    home_map = {"MORTGAGE": 0, "OTHER": 1, "OWN": 2, "RENT": 3}
    intent_map = {"DEBTCONSOLIDATION": 0, "EDUCATION": 1,
                  "HOMEIMPROVEMENT": 2, "MEDICAL": 3,
                  "PERSONAL": 4, "VENTURE": 5}
    grade_map = {"A": 0, "B": 1, "C": 2, "D": 3,
                 "E": 4, "F": 5, "G": 6}
    default_map = {"N": 0, "Y": 1}

    loan_to_income = loan_amnt / (person_income + 1)
    rate_x_amount = loan_int_rate * loan_amnt
    emp_to_age = person_emp_length / (person_age + 1)

    input_dict = {fn: 0 for fn in feature_names}
    input_dict.update({
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": home_map.get(person_home_ownership, 0),
        "person_emp_length": person_emp_length,
        "loan_intent": intent_map.get(loan_intent, 0),
        "loan_grade": grade_map.get(loan_grade, 0),
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": default_map.get(cb_default, 0),
        "cb_person_cred_hist_length": cb_cred_hist,
        "LOAN_TO_INCOME": loan_to_income,
        "RATE_X_AMOUNT": rate_x_amount,
        "EMP_TO_AGE": emp_to_age,
    })

    input_df = pd.DataFrame([input_dict])[feature_names]
    input_scaled = scaler.transform(input_df)

    st.markdown("---")
    if st.button("🔮 Assess Credit Risk", use_container_width=True):
        prob = model.predict_proba(input_scaled)[0][1]
        risk_level = ("🔴 HIGH RISK" if prob > 0.7 else
                      "🟡 MEDIUM RISK" if prob > 0.4 else "🟢 LOW RISK")
        decision = "❌ REJECT" if prob > 0.5 else "✅ APPROVE"

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Default Probability", f"{prob * 100:.1f}%")
        col_b.metric("Risk Level", risk_level)
        col_c.metric("Recommendation", decision)

        if prob > 0.7:
            st.error("This applicant has a very high default risk. "
                     "Loan application should be rejected or reviewed "
                     "with strict conditions.")
        elif prob > 0.4:
            st.warning("This applicant shows moderate risk. Consider "
                       "approving with a reduced loan amount or "
                       "higher interest rate.")
        else:
            st.success("This applicant is low risk. Loan can be "
                       "approved under standard conditions.")

# ============================================================
# PAGE 3 — SHAP EXPLAINABILITY
# ============================================================
elif page == "🧠 SHAP Explainability":
    st.title("🧠 Why Did the Model Make This Decision?")
    st.markdown("SHAP values explain which features most influenced "
                "the model's predictions.")
    st.markdown("---")

    st.info("Calculating SHAP values — this may take a few seconds...")

    df_shap = df.copy()

    # Encode categoricals
    le = LabelEncoder()
    for col in df_shap.select_dtypes(include="object").columns:
        df_shap[col] = le.fit_transform(df_shap[col].astype(str))

    # Add engineered features
    df_shap["LOAN_TO_INCOME"] = (df_shap["loan_amnt"] /
                                  (df_shap["person_income"] + 1))
    df_shap["RATE_X_AMOUNT"] = (df_shap["loan_int_rate"] *
                                 df_shap["loan_amnt"])
    df_shap["EMP_TO_AGE"] = (df_shap["person_emp_length"] /
                               (df_shap["person_age"] + 1))

    df_shap = df_shap.drop(columns=["loan_status"], errors="ignore")

    X_sample = df_shap[feature_names].sample(
        100, random_state=42).fillna(0)
    X_scaled_sample = scaler.transform(X_sample)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled_sample)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    st.subheader("📊 Feature Impact Summary")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_scaled_sample,
                      feature_names=feature_names,
                      plot_type="bar", show=False)
    st.pyplot(fig)
    plt.clf()

    st.subheader("🔍 Detailed SHAP Dot Plot")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_scaled_sample,
                      feature_names=feature_names, show=False)
    st.pyplot(fig2)
    plt.clf()

    st.markdown("---")
    st.markdown("""
    **How to read this chart:**
    - Features at the **top** have the most impact on predictions
    - **Red dots** = high feature value, **Blue dots** = low feature value
    - Dots on the **right** push the model toward predicting default
    - Dots on the **left** push the model toward predicting no default
    """)

# ============================================================
# PAGE 4 — BUSINESS RECOMMENDATIONS
# ============================================================
elif page == "📋 Business Recommendations":
    st.title("📋 Business Recommendations")
    st.markdown("Strategic actions based on credit risk analysis.")
    st.markdown("---")

    st.subheader("🔑 Key Findings")
    st.markdown("""
    - **~22% default rate** — roughly 1 in 5 loans ends in default
    - **Younger applicants (18–25)** carry the highest default risk
    - **Medical and venture loans** have the highest default rates by intent
    - **Loan percent of income** is the strongest predictor of default
    - **High interest rates combined with large loan amounts** significantly
      increase default risk
    """)

    st.markdown("---")
    st.subheader("✅ Recommended Actions")

    col1, col2 = st.columns(2)
    with col1:
        st.success("**1. Cap loan-to-income ratio at 40%**\n\n"
                   "Applicants whose loan repayment exceeds 40% of their "
                   "monthly income should be flagged for manual review "
                   "before approval.")
        st.success("**2. Tighten criteria for young applicants**\n\n"
                   "Applicants aged 18–25 with short employment history "
                   "should face stricter income verification and lower "
                   "initial loan limits.")
    with col2:
        st.success("**3. Offer tiered loan products**\n\n"
                   "Medium-risk applicants (40–70% probability) can be "
                   "offered smaller loan amounts at higher interest rates "
                   "rather than outright rejection.")
        st.success("**4. Use this model in the approval pipeline**\n\n"
                   "Run predictions on all applications automatically. "
                   "Any applicant scoring above 50% default probability "
                   "should be flagged for human review.")

    st.markdown("---")
    st.subheader("📈 Estimated Business Impact")
    impact = pd.DataFrame({
        "Action": ["Loan-to-income cap",
                   "Young applicant screening",
                   "Tiered loan products",
                   "Model in approval pipeline"],
        "Expected Default Reduction": ["20-25%", "15-20%",
                                        "10-15%", "25-30%"],
        "Implementation Difficulty": ["Low", "Low", "Medium", "Medium"]
    })
    st.dataframe(impact, use_container_width=True)