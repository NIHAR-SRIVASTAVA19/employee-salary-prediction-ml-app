import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Title and subtitle
st.markdown("<h1 style='text-align: center;'>📊 Model Performance Summary</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>A quick glance at how different models performed on your dataset.</p>", unsafe_allow_html=True)
st.markdown("----")

# Load the saved model performance CSV
csv_path = "C:\My_Place\Employee_Salary_Prediction-Using-ML_Algos\employee-salary-prediction-ml-app\streamlit_app\pages\model_performance.csv"
try:
    df = pd.read_csv(csv_path)

    # Show top model
    best_model = df.iloc[0]
    st.markdown("### 🏆 Best Performing Model")
    st.success(f"**{best_model['Model']}** achieved the highest accuracy of **{best_model['Accuracy']:.4f}**.")

    # Display Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{best_model['Accuracy']:.4f}")
    col2.metric("Precision", f"{best_model['Precision']:.4f}")
    col3.metric("Recall", f"{best_model['Recall']:.4f}")
    col4.metric("F1 Score", f"{best_model['F1-score']:.4f}")

    st.markdown("----")

    # Bar Chart of all models
    st.subheader("📈 Accuracy Comparison of All Models")
    fig, ax = plt.subplots()
    sns.barplot(data=df, y="Model", x="Accuracy", palette="Blues_d", ax=ax)
    ax.set_title("Model Accuracy Comparison")
    st.pyplot(fig)

    st.markdown("----")

    # Performance Table
    st.subheader("📋 Full Performance Table")
    st.dataframe(df, use_container_width=True)

    st.markdown("----")

except FileNotFoundError:
    st.error("⚠️ model_performance.csv not found in `pages` folder.")

st.markdown("## 🔍 Confusion Matrix")

try:
    cm = np.load("C:\My_Place\Employee_Salary_Prediction-Using-ML_Algos\employee-salary-prediction-ml-app\streamlit_app\pages\confusion_matrix.npy")  # File is in the same folder as performance.py
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

except FileNotFoundError:
    st.warning("⚠️ confusion_matrix.npy not found in `pages` directory.")
