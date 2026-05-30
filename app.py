import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Loan Eligibility Prediction",
    page_icon="🏦",
    layout="centered"
)

# Application Header
st.title("🏦 Loan Eligibility Prediction App")
st.write("Enter the applicant's details below to evaluate loan eligibility in real-time.")

# Path to the serialized model file
MODEL_PATH = "loan_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.warning(f"⚠️ Model file `{MODEL_PATH}` not found in the current directory. Please ensure you have run your `model.py` script first to generate it.")
else:
    # Load the trained Random Forest model
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
        
    st.markdown("### 📋 Applicant Information Form")
    
    # Create two columns for a clean and responsive form layout
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital Status", ["Married (Yes)", "Unmarried (No)"])
        dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed?", ["No", "Yes"])
        property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

    with col2:
        applicant_income = st.number_input("Applicant Monthly Income ($)", min_value=0, value=5000, step=100)
        coapplicant_income = st.number_input("Co-applicant Monthly Income ($)", min_value=0, value=2000, step=100)
        loan_amount = st.number_input("Loan Amount (in Thousands, e.g., 150 = $150,000)", min_value=1, value=150, step=10)
        loan_term = st.number_input("Loan Amount Term (in Months, standard is 360)", min_value=12, value=360, step=12)
        credit_history = st.selectbox(
            "Credit History Status", 
            options=[1.0, 0.0], 
            format_func=lambda x: "Good Credit History (1.0)" if x == 1.0 else "Poor/No Credit History (0.0)"
        )

    # --- ENCODING LAYER ---
    # Values map alphabetically to replicate LabelEncoder().fit_transform() logic from model.py
    gender_encoded = 1 if gender == "Male" else 0
    married_encoded = 1 if "Yes" in married else 0
    dependents_encoded = int(dependents[0]) if dependents != "3+" else 3
    education_encoded = 0 if education == "Graduate" else 1
    self_employed_encoded = 1 if self_employed == "Yes" else 0
    
    property_maps = {"Rural": 0, "Semiurban": 1, "Urban": 2}
    property_encoded = property_maps[property_area]

    # --- FEATURE ALIGNMENT ---
    # Features sequence matches X.columns exactly as expected by your model
    feature_names = [
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
        'Credit_History', 'Property_Area'
    ]
    
    features_list = [
        gender_encoded,
        married_encoded,
        dependents_encoded,
        education_encoded,
        self_employed_encoded,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_encoded
    ]
    
    # Pack parameters into a single-row DataFrame
    new_applicant = pd.DataFrame([features_list], columns=feature_names)
    
    st.markdown("---")
    
    # Prediction Button execution
    if st.button("🚀 Evaluate Loan Application", type="primary"):
        prediction = model.predict(new_applicant)
        
        # Pull model output probabilities for extra metric visibility
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(new_applicant)[0]
            confidence = probabilities[prediction[0]] * 100
            confidence_str = f" (Confidence: {confidence:.2f}%)"
        else:
            confidence_str = ""
            
        st.markdown("### 📊 Decision Result")
        if prediction[0] == 1:
            st.success(f"🎉 **Congratulations! The Loan Application is APPROVED.**{confidence_str}")
            st.balloons()
        else:
            st.error(f"❌ **Sorry, the Loan Application has been REJECTED.**{confidence_str}")

    # Expandable summary detailing encoded inputs
    with st.expander("ℹ️ View Model Feature Encodings"):
        st.write("Behind the scenes, your custom selections map to the exact integer representations used during model training:")
        st.code('''
- Gender: Female -> 0, Male -> 1
- Married: No -> 0, Yes -> 1
- Dependents: 0 -> 0, 1 -> 1, 2 -> 2, 3+ -> 3
- Education: Graduate -> 0, Not Graduate -> 1
- Self_Employed: No -> 0, Yes -> 1
- Property_Area: Rural -> 0, Semiurban -> 1, Urban -> 2
        ''')
        
        