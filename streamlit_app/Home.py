import streamlit as st

st.set_page_config(page_title="Employee Salary Prediction App", layout="wide")

# === Custom Styles ===
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #1E1E1E;
        padding: 1.5rem;
    }
    [data-testid="stSidebarNav"]::before {
        content: "📌NAVIGATION";
        margin-left: 15px;
        font-size: 1.2rem;
        font-weight: bold;
        color: #ffffff;
        display: block;
        margin-top: 20px;
    }
    .main-title {
        margin-top: -50px;
        text-align: center;
        font-size: 3em;
        color: #00BFFF;
        font-weight: 900;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #BBBBBB;
        margin-bottom: 40px;
    }
    .card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 2px solid #444;
        transition: 0.5s;
    }
    .card:hover {
        border-color: #4CAF50;
        transform: scale(1.02);
    }
    .emoji {
        font-size: 2rem;
    }
    .card-title {
        font-size: 1.5em;
        font-weight: bold;
        margin: 10px 0;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)
# === Custom Sidebar Content ===
st.sidebar.markdown("---")

# === Title & Subtitle ===
st.markdown("<div class='main-title'>💼 Employee Salary Prediction App</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict whether an employee earns >50K or <=50k using ML models.</div>", unsafe_allow_html=True)
st.markdown("""
    <div class='subtitle'>
        A machine learning-powered tool that predicts if an employee earns above or below 50K.<br>
        Simply explore models, understand important features, and try it yourself!
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

# === Info Cards ===
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='card'><div class='emoji'>📊</div><div class='card-title'>Model Performance</div><p>Compare how various models performed on the dataset.</p></div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'><div class='emoji'>📈</div><div class='card-title'>Feature Importance</div><p>Understand which features influenced the model the most.</p></div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='card'><div class='emoji'>🧠</div><div class='card-title'>Salary Prediction</div><p>Test the trained model with your own input values.</p></div>", unsafe_allow_html=True)

st.markdown("---")

# === Footer Info ===
st.info("ℹ️ Use the left sidebar to navigate between sections.")
