import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Churn Guardian AI", page_icon="🛡️", layout="wide")
st.title("🛡️ Telco Customer Churn Prevention System")
st.markdown("An AI-powered dashboard to predict, explain, and prevent customer churn.")

# ==========================================
# DATA LOADING & CACHING
# ==========================================
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    customer_ids = df['customerID']
    df = df.drop('customerID', axis=1)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df, customer_ids

@st.cache_resource
def train_pipeline(df):
    """Engineers features, builds pipeline, and trains model instantly."""
    df_eng = df.copy()
    
    # 1. Feature Engineering (From your notebook)
    no_service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']
    for col in no_service_cols:
        df_eng[col] = df_eng[col].replace({'No internet service': 'No', 'No phone service': 'No'})
        
    df_eng['Spending_Intensity'] = df_eng['MonthlyCharges'] / (df_eng['tenure'] + 1)
    contract_order = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    df_eng['Contract_Ordinal'] = df_eng['Contract'].map(contract_order)
    df_eng['MTM_ElectronicCheck'] = ((df_eng['Contract'] == 'Month-to-month') & (df_eng['PaymentMethod'] == 'Electronic check')).astype(int)
    
    df_eng = df_eng.drop(['TotalCharges', 'Contract'], axis=1)
    
    X = df_eng.drop('Churn', axis=1)
    y = df_eng['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Pipeline Setup
    cat_features = X_train.select_dtypes(include=['object']).columns.tolist()
    num_features = X_train.select_dtypes(exclude=['object']).columns.tolist()
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_features)
    ])
    
    # 3. Model Training (Using optimal parameters)
    model = LGBMClassifier(
        class_weight='balanced', random_state=42, force_row_wise=True,
        n_estimators=200, learning_rate=0.05, max_depth=5, num_leaves=31
    )
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train, y_train)
    
    # 4. Extract SHAP Explainer
    X_test_trans = pipeline.named_steps['preprocessor'].transform(X_test)
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    X_test_shap = pd.DataFrame(X_test_trans, columns=feature_names)
    
    explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
    shap_values = explainer.shap_values(X_test_shap)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        
    # Evaluate
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "pr_auc": average_precision_score(y_test, y_prob),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    
    return pipeline, explainer, X_test_shap, shap_values, X, metrics

# Load everything
with st.spinner("Initializing AI Model..."):
    raw_df, customer_ids = load_and_clean_data()
    pipeline, explainer, X_test_shap, shap_values, X_engineered, metrics = train_pipeline(raw_df)

# ==========================================
# UI NAVIGATION
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Executive Dashboard", "🎯 Predict & Explain (SHAP)", "🧠 Global Drivers"])

# --- TAB 1: DASHBOARD ---
with tab1:
    st.header("Overall Business Health")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{len(raw_df):,}")
    col2.metric("Overall Churn Rate", f"{(raw_df['Churn'].mean() * 100):.1f}%")
    col3.metric("Model PR-AUC (Precision-Recall)", f"{metrics['pr_auc']:.3f}")
    
    st.divider()
    st.subheader("High-Risk Segments (Exploratory)")
    
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(data=raw_df, x='Contract', y='Churn', errorbar=None, palette=['#E74C3C', '#2ECC71', '#3498DB'], ax=ax)
        ax.set_title("Churn Rate by Contract Type")
        ax.set_ylabel("Churn Probability")
        st.pyplot(fig)
        
    with c2:
        fig2, ax2 = plt.subplots(figsize=(6,4))
        sns.kdeplot(data=raw_df[raw_df['Churn'] == 0], x='tenure', fill=True, color='#2ECC71', label='Retained', ax=ax2)
        sns.kdeplot(data=raw_df[raw_df['Churn'] == 1], x='tenure', fill=True, color='#E74C3C', label='Churned', ax=ax2)
        ax2.set_title("Tenure Distribution by Churn Status")
        ax2.legend()
        st.pyplot(fig2)

# --- TAB 2: PREDICTION ENGINE ---
with tab2:
    st.header("Individual Customer Analysis")
    st.markdown("Adjust the sliders below to simulate a customer's profile and see how the AI evaluates their churn risk in real-time.")
    
    col_input, col_output = st.columns([1, 2])
    
    with col_input:
        st.subheader("Customer Profile")
        in_tenure = st.slider("Tenure (Months)", 0, 72, 12)
        in_monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 75.0)
        in_contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        in_payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        in_internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        in_senior = st.radio("Senior Citizen?", [0, 1], index=0)
        
        # Build dummy dataframe for prediction (filling defaults for the rest)
        user_data = raw_df.drop('Churn', axis=1).iloc[0:1].copy()
        user_data['tenure'] = in_tenure
        user_data['MonthlyCharges'] = in_monthly
        user_data['Contract'] = in_contract
        user_data['PaymentMethod'] = in_payment
        user_data['InternetService'] = in_internet
        user_data['SeniorCitizen'] = in_senior
        
        # Apply manual feature engineering so the pipeline accepts it
        user_data['Spending_Intensity'] = user_data['MonthlyCharges'] / (user_data['tenure'] + 1)
        contract_order = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        user_data['Contract_Ordinal'] = user_data['Contract'].map(contract_order)
        user_data['MTM_ElectronicCheck'] = int(in_contract == 'Month-to-month' and in_payment == 'Electronic check')
        user_data = user_data.drop(['TotalCharges', 'Contract'], axis=1)

    with col_output:
        st.subheader("AI Prediction & Explanation")
        
        # Predict
        prob = pipeline.predict_proba(user_data)[0][1]
        
        # Threshold logic (Using the 0.53 threshold you found optimal)
        is_churn = prob >= 0.53
        
        if is_churn:
            st.error(f"🚨 **HIGH RISK**: {prob*100:.1f}% probability of churn.")
        else:
            st.success(f"✅ **LOW RISK**: {prob*100:.1f}% probability of churn.")
            
        st.markdown("**Why did the AI make this decision? (SHAP Waterfall)**")
        
        # Process the single row for SHAP
        user_data_trans = pipeline.named_steps['preprocessor'].transform(user_data)
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        
        # Calculate local SHAP values
        single_shap = explainer.shap_values(user_data_trans)
        if isinstance(single_shap, list): single_shap = single_shap[1]
            
        # Plot Waterfall
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        shap.waterfall_plot(shap.Explanation(values=single_shap[0], 
                                             base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value, 
                                             data=user_data_trans[0], 
                                             feature_names=feature_names), 
                            show=False)
        st.pyplot(fig3)
        st.caption("*Red bars push the customer towards churning. Blue bars push them towards staying.*")

# --- TAB 3: GLOBAL INSIGHTS ---
with tab3:
    st.header("What drives churn across the entire business?")
    st.markdown("The SHAP Summary plot evaluates every customer in the test set to determine which features have the highest aggregate impact on churn, and in what direction.")
    
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_shap, show=False)
    st.pyplot(fig4)
    
    st.info("""
    **How to read this chart:**
    * Features at the **top** are the most important.
    * A **Red dot** means the original feature value was HIGH. A **Blue dot** means it was LOW.
    * Dots on the **Right** (positive SHAP value) increased churn risk. Dots on the **Left** decreased it.
    * *Example:* Low values (blue) for `Contract_Ordinal` (meaning Month-to-Month) strongly push predictions to the right (higher churn).
    """)
