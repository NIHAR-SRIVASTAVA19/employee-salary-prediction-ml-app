import streamlit as st
import pandas as pd
import pickle
import numpy as np
import joblib
import os


# Mapping for education levels to numerical values
education_mapping = {
    "7th-8th Grade": 4,
    '9th Grade': 5,
    '10th Grade': 6,
    '11th Grade': 7,
    '12th Grade': 8,
    'High School Graduate': 9,
    'Some College': 10,
    'Associate Degree (Vocational)': 11,
    'Associate Degree (Academic)': 12,
    "Bachelor's Degree": 13,
    "Master's Degree": 14,
    'Professional School': 15,
    'Doctorate': 16
}

workclass_mapping = {
    "Private Sector": "Private",
    "Self-employed (Not Incorporated)": "Self-emp-not-inc",
    "Self-employed (Incorporated)": "Self-emp-inc",
    "Federal Government": "Federal-gov",
    "Local Government": "Local-gov",
    "State Government": "State-gov",
    "Others": "NotListed"
}

marital_status_mapping = {
    "Married (Spouse Present)": "Married-civ-spouse",
    "Married (Spouse Absent)": "Married-spouse-absent",
    "Never Married": "Never-married",
    "Divorced": "Divorced",
    "Separated": "Separated",
    "Widowed": "Widowed",
    "Others": "Others"
}

occupation_mapping = {
    "Tech Support": "Tech-support",
    "Skilled Crafts & Repairs": "Craft-repair",
    "Personal & Other Services": "Other-service",
    "Sales": "Sales",
    "Executive / Managerial Roles": "Exec-managerial",
    "Professional (e.g., doctors, engineers)": "Prof-specialty",
    "Manual Labor & Cleaning": "Handlers-cleaners",
    "Machine Operators & Inspectors": "Machine-op-inspct",
    "Administrative & Clerical": "Adm-clerical",
    "Farming & Fishing": "Farming-fishing",
    "Transportation & Moving": "Transport-moving",
    "Private Household Services": "Priv-house-serv",
    "Security & Law Enforcement": "Protective-serv",
    "Military / Armed Forces": "Armed-Forces",
    "Not Listed": "NotListed"
}

race_mapping = {
    "White": "White",
    "Black or African-American": "Black",
    "Asian or Pacific Islander": "Asian-Pac-Islander",
    "American Indian or Alaska Native": "Amer-Indian-Eskimo",
    "Other / Not Specified": "Other"
}
# Get path of the current file (prediction.py)
CURRENT_DIR = os.path.dirname(__file__)

# Navigate two levels up to reach the root folder
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))

# Build correct path to the model
model_path = os.path.join(BASE_DIR, 'models', 'gb_pipeline.pkl')


try:
    pipeline = joblib.load(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Unexpected error while loading model: {e}")

st.title("🔮 Income Prediction")
st.markdown("Fill in the details below to predict whether the income is **>50K** or **<=50K**.")

# Input form
with st.form("prediction_form"):
    age = st.slider("Age", 18, 90, 30, help="Select your current age.")
    education = st.selectbox("Education",list(education_mapping.keys()),help="Highest education level you have completed.")

    gender = st.radio("Gender", ["Male", "Female"],help="Select your gender (used for statistical learning).")

    workclass = st.selectbox("Workclass",list(workclass_mapping.keys()),help="Type of employer or work sector.")

    marital_status = st.selectbox("Marital Status",list(marital_status_mapping.keys()),help="Your current marital status.")

    occupation = st.selectbox("Occupation",list(occupation_mapping.keys()),help="Your current occupation.")

    race = st.selectbox("Race", list(race_mapping.keys()
),help="Choose your race/ethnic group.")

    native_country = st.selectbox("Native Country", ["United-States", "Mexico", "Philippines", "Germany", "Canada", "India","Puerto-Rico","El-Salvador","Cuba","England","China","South","Jamaica","Italy","Dominican-Republic","NotListed/Others"],help="Country you were born in or primarily identify with.")

    hours = st.slider("Hours per week", 1, 99, 40,help="Average number of hours you work per week.")

    capital_gain = st.number_input("Capital Gain", min_value=0,help="Investment gains (if any). Usually 0 for most people.")

    capital_loss = st.number_input("Capital Loss", min_value=0,help="Investment losses (if any). Usually 0 for most people.")

    submitted = st.form_submit_button("Predict")




if submitted:
    # Convert gender to binary if needed (you had used LabelEncoder before)
    gender_binary = 1 if gender == "Male" else 0
    # Ensure the selected education is in the mapping
    edu_num = education_mapping[education]
    # Mapping for occupation
    occup=occupation if occupation not in ["Armed-Forces", "Priv-house-serv"] else "Others"
    # Backend mapping
    if native_country == "Other / Not Listed":
        native_country = "Other"

    # Create input DataFrame
    input_df = pd.DataFrame([{
        'age': age,
        'educational-num': edu_num,
        'gender': gender_binary,
        'workclass': workclass,
        'marital-status': marital_status,
        'occupation': occup,
        'race': race,
        'native-country': native_country,
        'hours-per-week': hours,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss
    }])
    # Predict
    prediction = pipeline.predict(input_df)[0]

    # Confidence
    proba = pipeline.predict_proba(input_df)[0]
    confidence = max(proba) * 100

    prediction_label = ">50K" if prediction == 1 else "<=50K"

    st.markdown(f"""
    ### 🔮 Prediction: `{prediction_label}`
    #### 📈 Confidence: `{confidence:.2f}%`
    """)



