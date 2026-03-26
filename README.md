 **Loan Production & Eligibility App**

A comprehensive **End-to-End Machine Learning Pipeline** that predicts loan eligibility using a hybrid approach: a traditional **Rule-Based Risk Engine** and a **Random Forest Classifier**. Built with Python and deployed via an interactive Streamlit dashboard.

## **Features**

* **Synthetic Data Generation:** Custom script to generate realistic financial datasets with 500+ records.  
* **Hybrid Decision Engine\* Rule-Based:** Uses weighted logic (Credit Score 50%, DTI 30%, EMI Ratio 20%) to calculate a real-time risk score.  
* **ML-Based:** A Random Forest model trained on scaled financial features to provide probability-based predictions.  
* **Interactive UI:** A polished Streamlit interface featuring a Plotly Gauge chart for visual risk assessment.  
* **Data Persistence:** Ability to save new applications directly to a CSV file to expand the training dataset.

##  **Technical Stack**

* **Frontend:** Streamlit, Plotly (Data Visualization)  
* **Data Processing:** Pandas, NumPy  
* **Machine Learning:** Scikit-Learn (Random Forest, StandardScaler)  
* **Model Storage:** Pickle

##  **Project Structure**

├── generate\_synthetic\_loans,py   \- Script to generate synthetic loan data  
├── train\_loan\_model.py      \- ML pipeline (Preprocessing, Training, & Export)  
├── loan.py              \- Main Streamlit Dashboard code  
├── loan\_applications.csv \- The dataset (Generated & Appended)  
├── loan\_model.pkl      \- Trained Random Forest model  
└── scaler.pkl          \- Saved StandardScaler instance

##  **How It Works**

### **1\. Risk Calculation Logic**

The app evaluates users based on a 100-point scale:

RiskScore \= ( (CreditWeight \* 0.5) \+ (1 \- DTI) \* 0.3 \+ (1 \- EMIRatio) \* 0.2 ) 

### **2\. Machine Learning Pipeline**

The model considers 8 distinct features:

* Monthly Income, Expenses, and EMIs.  
* Credit Score (300-900).  
* Loan Amount Requested & Tenure.  
* Calculated Ratios: **DTI** (Debt-to-Income) and **EMI Ratio**.

##  **Usage**

1. Enter your financial details in the input fields.  
2. View your **Eligibility Gauge** (Red: Rejected, Yellow: Borderline, Green: Approved).  
3. Compare the manual risk score with the **ML Model's Prediction Probabilities**.  
4. Click **"Save Application"** to log the data for future analysis.

### 
