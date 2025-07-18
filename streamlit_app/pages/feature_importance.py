import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the page title
st.title("🔍 Feature Importance Analysis")

BASE_DIR = os.path.dirname(__file__)
csv_path = os.path.join(BASE_DIR, 'feature_importance.csv')

# === Friendly mapping ===
friendly_feature_names = {
    'cat__marital-status_Married-civ-spouse': 'Marital Status: Married',
    'num__capital-gain': 'Capital Gain',
    'num__educational-num': 'Education',
    'num__capital-loss': 'Capital Loss',
    'num__age': 'Age',
    'num__hours-per-week': 'Hours per Week',
    'cat__occupation_Exec-managerial': 'Occupation: Executive Managerial',
    'cat__occupation_Prof-specialty': 'Occupation: Professional Specialty',
    'cat__occupation_Other-service': 'Occupation: Other Service',
    'cat__workclass_Self-emp-not-inc': 'Workclass: Self-emp (Not Inc.)',
    'cat__occupation_Farming-fishing': 'Occupation: Farming/Fishing',
    'num__gender': 'Gender',
    'cat__workclass_Federal-gov': 'Workclass: Federal Gov',
    'cat__occupation_Handlers-cleaners': 'Occupation: Handlers/Cleaners',
    'cat__occupation_Tech-support': 'Occupation: Tech Support',
    'cat__workclass_Local-gov': 'Workclass: Local Gov',
    'cat__occupation_Sales': 'Occupation: Sales',
    'cat__occupation_Machine-op-inspct': 'Occupation: Machine Operator',
    'cat__occupation_Protective-serv': 'Occupation: Protective Services',
    'cat__workclass_Private': 'Workclass: Private'
}

try:
    # Load the feature importance data
    feature_df = pd.read_csv(csv_path)

    # Map to user-friendly names
    feature_df['Feature'] = feature_df['Feature'].map(lambda x: friendly_feature_names.get(x, x))

    st.markdown("""
    This section highlights the **most important features** used by the selected model (Gradient Boosting) 
    to make predictions. Understanding these can help in gaining insights into what factors influence income classification.
    """)
    st.markdown("---")

    # Display the top 10 in a table
    st.subheader("📊 Top 10 Important Features")
    st.dataframe(feature_df.head(10), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Plot top 20 as bar chart
    st.subheader("📈 Top 20 Features - Bar Chart")
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_df.head(20), y='Feature', x='Importance', hue='Feature', legend=False)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title("Top 20 Feature Importances")
    st.pyplot(plt.gcf())  # Show the current figure in Streamlit

except FileNotFoundError:
    st.error("⚠️ Could not find 'feature_importance.csv'. Make sure it's placed in `streamlit_app/pages/`.")
