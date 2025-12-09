import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import pandas as pd
import pickle
import numpy as np

MODEL_PATH = Path("loan_model.pkl")
SCALER_PATH = Path("scaler.pkl")

model = None
scaler = None

if MODEL_PATH.exists() and SCALER_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    st.sidebar.success("ML model loaded ✅")
else:
    st.sidebar.warning("ML model not found. Using only rule-based logic.")

st.title("LOAN ELIGIBILITY CRITERIA")
st.write("Lets Check Your Eligibility")
st.divider()

st.header("Lets Calculate!!")
name = st.text_input("Enter Your Full Name: ")
income = st.number_input("Enter Your Monthly Income (In Lakhs)(₹): ", min_value= 0 , max_value=10000000000 , value=200000)
expenses = st.number_input("Enter Your Monthly Expenses(In Lakhs)(₹): ", min_value=0 , max_value=10000000000, value=200000)
credit_score = st.number_input("Enter Your Credit Score: ", min_value=300, max_value=900, value=500)
emis = st.number_input("Enter Your Monthly Emi's(In Lakhs)(₹): ", min_value=0,max_value=10000000000,value=200000)
loan_req = st.number_input("Enter Your Amount Requested(In Lakhs)(₹): ", min_value=0,max_value=10000000000,value=200000)
years = st.number_input("Enter Your Tenure(In Years): ",min_value=0,max_value=20,value=5)

# calculation time
dti = (expenses + emis) / income
emi_ratio = emis / income 

if credit_score >= 750:
    credit_weight = 1
elif credit_score >= 650 and credit_score < 750:
    credit_weight = 0.6
else:
    credit_weight = 0.2

risk_score = (
    credit_weight * 0.5 +     
    (1 - dti) * 0.3 +           
    (1 - emi_ratio) * 0.2     
) * 100
risk_score = max(0.0, min(100.0, risk_score))

st.divider()

st.header("Results!!")

st.subheader("Here You Go!")

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=risk_score,
    number={'valueformat': '.1f'},
    gauge={
        'axis': {'range': [0, 100]},  
        'bar': {'color': 'blue'},   
        'steps': [
            {'range': [0, 50], 'color': "red"},   
            {'range': [50, 70], 'color': "yellow"},  
            {'range': [70, 100], 'color': "lightgreen"},       
           
        ],
    }
))

st.plotly_chart(fig, use_container_width=True)

if risk_score >= 70:
    st.success("You are Eligible for loan")
    decision = "Approved"
elif risk_score >= 50 and risk_score < 70:
    st.warning("Maybe You can get a Loan ")
    decision = "Borderline"
else:
    st.error("We are Sorry")
    decision = "Rejected"

ml_decision = None
ml_proba_dict = None

if model is not None and scaler is not None and income > 0:

    features = np.array([[
        income,
        expenses,
        credit_score,
        emis,
        loan_req,
        years,
        dti,
        emi_ratio
    ]])

    features_scaled = scaler.transform(features)

    proba = model.predict_proba(features_scaled)[0]
    pred_class = model.predict(features_scaled)[0]   

    idx_to_label = {0: "Rejected", 1: "Borderline", 2: "Approved"}
    ml_decision = idx_to_label[int(pred_class)]

    ml_proba_dict = {
        "Rejected":   proba[0],
        "Borderline": proba[1],
        "Approved":   proba[2],
    }

st.divider()
st.header("ML Model Prediction (from trained Random Forest)")

if ml_decision is None:
    st.info("ML model not available or invalid inputs. Showing only rule-based result.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ML Decision")
        if ml_decision == "Approved":
            st.success(f"ML Model says: {ml_decision}")
        elif ml_decision == "Borderline":
            st.warning(f"ML Model says: {ml_decision}")
        else:
            st.error(f"ML Model says: {ml_decision}")

    with col2:
        st.subheader("Class Probabilities")
        st.write(f"Rejected:   {ml_proba_dict['Rejected']*100:.1f}%")
        st.write(f"Borderline: {ml_proba_dict['Borderline']*100:.1f}%")
        st.write(f"Approved:   {ml_proba_dict['Approved']*100:.1f}%")


st.write(f"**Calculated risk score:** {risk_score:.1f}")
st.write(f"**Debt-to-Income ratio (DTI):** {dti:.2f}")
st.write(f"**EMI / Income ratio:** {emi_ratio:.2f}")


st.divider()

st.subheader("Save this application")

csv_path = Path("loan_applications.csv")

if st.button("Save Application"):
    if name.strip() == "":
        st.error("Please enter your name before saving.")
    else:
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": name,
            "income": income,
            "expenses": expenses,
            "credit_score": credit_score,
            "emis": emis,
            "loan_requested": loan_req,
            "tenure_years": years,
            "dti": dti,
            "emi_ratio": emi_ratio,
            "risk_score": risk_score,
            "decision": decision,
        }

        if csv_path.exists():
            old_df = pd.read_csv(csv_path)
            new_df = pd.concat([old_df, pd.DataFrame([row])], ignore_index=True)
        else:
            new_df = pd.DataFrame([row])

        new_df.to_csv(csv_path, index=False)
        st.success("Application saved to loan_applications.csv")
        

st.subheader("Made with ❤️ from Shivansh")















