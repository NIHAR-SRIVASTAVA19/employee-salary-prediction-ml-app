import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the page title
st.title("🔍 Feature Importance Analysis")

BASE_DIR = os.path.dirname(__file__)
csv_path = os.path.join(BASE_DIR, 'feature_importance.csv')



try:
    # Load the feature importance data
    feature_df = pd.read_csv(csv_path)

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
    sns.barplot(data=feature_df.head(20), y='Feature', x='Importance',hue="Feature", legend=False)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title("Top 20 Feature Importances")
    st.pyplot(plt.gcf())  # Pass current figure to Streamlit

except FileNotFoundError:
    st.error("⚠️ Could not find 'feature_importance.csv'. Make sure it's placed in `streamlit_app/pages/`.")
