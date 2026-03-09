import streamlit as st
import pandas as pd
import joblib
import io

# --- 1. SETTINGS & ASSETS ---
st.set_page_config(page_title="ChurnGuard AI", layout="wide")

# Load your model (adjust path if necessary)
@st.cache_resource
def load_model():
    return joblib.load('random_forest_model.pkl')

model = load_model()

# --- 2. HEADER ---
st.title("🛡️ ChurnGuard AI: Predictive Analytics")
st.markdown("""
    Predict customer attrition with high precision using our tuned Random Forest engine.
    Choose between **Single Prediction** for individual cases or **Batch Inference** for CSV uploads.
""")
st.divider()

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Select Prediction Mode", ["Single Prediction", "Batch Prediction"])

# Define the feature columns exactly as they appear in your X_train
FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
    'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
    'MonthlyCharges', 'TotalCharges', 'InternetService_Fiber optic', 
    'InternetService_No', 'PaymentMethod_Credit_card', 'PaymentMethod_Electronic', 
    'PaymentMethod_Mailed'
]

# --- 4. SINGLE PREDICTION ---
if app_mode == "Single Prediction":
    st.subheader("👤 Individual Customer Profile")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender (0=M, 1=F)", [0, 1])
        senior = st.selectbox("Senior Citizen", [0, 1])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        contract = st.selectbox("Contract Type (0, 1, 2)", [0, 1, 2])
        
    with col2:
        monthly = st.number_input("Monthly Charges", value=50.0)
        total = st.number_input("Total Charges", value=500.0)
        paperless = st.selectbox("Paperless Billing", [0, 1])
        internet_fiber = st.selectbox("Fiber Optic Internet", [0, 1])
        
    with col3:
        online_sec = st.selectbox("Online Security", [0, 1])
        tech_supp = st.selectbox("Tech Support", [0, 1])
        streaming_tv = st.selectbox("Streaming TV", [0, 1])
        payment_elec = st.selectbox("Electronic Check", [0, 1])

    # Mapping inputs to the full feature list (filling others with 0 for simplicity)
    input_dict = {feat: 0 for feat in FEATURES}
    input_dict.update({
        'gender': gender, 'SeniorCitizen': senior, 'tenure': tenure,
        'Contract': contract, 'MonthlyCharges': monthly, 'TotalCharges': total,
        'PaperlessBilling': paperless, 'InternetService_Fiber optic': internet_fiber,
        'OnlineSecurity': online_sec, 'TechSupport': tech_supp, 
        'StreamingTV': streaming_tv, 'PaymentMethod_Electronic': payment_elec
    })
    
    input_df = pd.DataFrame([input_dict])

    if st.button("Analyze Risk"):
        prediction = model.predict(input_df)
        prob = model.predict_proba(input_df)[0][1]
        
        if prediction[0] == 1:
            st.error(f"⚠️ High Risk! Probability of Churn: {prob:.2%}")
        else:
            st.success(f"✅ Low Risk. Probability of Churn: {prob:.2%}")

# --- 5. BATCH PREDICTION ---
else:
    st.subheader("📊 Batch Inference")
    st.info("Upload a CSV file containing customer data. Ensure columns match the training format.")
    
    # Provide sample download
    st.download_button("Download Template (sample.csv)", 
                       data=pd.read_csv("main/sample.csv").to_csv(index=False), 
                       file_name="sample_template.csv")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        
        # Ensure only columns model knows are passed
        model_input = data[FEATURES]
        
        results = model.predict(model_input)
        results_prob = model.predict_proba(model_input)[:, 1]
        
        data['Churn_Prediction'] = results
        data['Churn_Probability'] = results_prob
        data['Status'] = data['Churn_Prediction'].apply(lambda x: "🚨 Churn" if x == 1 else "🟢 Stay")

        st.write("### Prediction Results")
        st.dataframe(data.style.background_gradient(subset=['Churn_Probability'], cmap='Reds'))
        
        # Download Results
        csv_result = data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Results", data=csv_result, file_name="churn_results.csv")

# --- 6. RESTART ---
if st.sidebar.button("🔄 Restart Session"):
    st.rerun()