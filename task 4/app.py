import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Page Configuration ---
st.set_page_config(page_title="HeartCare Luxe AI", page_icon="💖", layout="wide")

# --- Custom CSS for Pink, Purple & Glassmorphism ---
st.markdown("""
    <style>
    /* Main Background with Gradient */
    .stApp {
        background: linear-gradient(135deg, #1a0a2e 0%, #4a1d5a 50%, #2e0a2e 100%);
        color: #fce4ec;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(74, 29, 90, 0.5) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Titles and Headers */
    h1, h2, h3, p {
        color: #f8bbd0 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    /* Shiny Neon Button */
    .stButton>button {
        width: 100%;
        border-radius: 30px;
        height: 3.5em;
        background: linear-gradient(45deg, #d81b60, #8e24aa);
        color: white !important;
        font-weight: bold;
        border: none;
        box-shadow: 0 0 20px rgba(216, 27, 96, 0.6);
        transition: all 0.4s ease;
        font-size: 18px;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 35px rgba(216, 27, 96, 0.9);
        color: #fff !important;
    }

    /* Glass Effect for Cards */
    div[data-testid="column"] {
        background: rgba(255, 255, 255, 0.05);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
    }

    /* Dataframe & Table Styling */
    .stDataFrame {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 15px;
    }
    
    /* Custom Metric Styling */
    [data-testid="stMetricValue"] {
        color: #f48fb1 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header Section ---
st.markdown("<h1 style='text-align: center; font-size: 50px;'>💖 HeartCare Luxe AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.8;'>Professional Cardiac Analysis with a touch of Elegance</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Data Loading ---
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

try:
    df = load_data()
except:
    st.error("Please upload 'heart.csv' to the directory.")
    st.stop()

# --- Sidebar Inputs (Pink/Purple Theme) ---
st.sidebar.markdown("<h2 style='color: #f48fb1;'>✨ Patient Profile</h2>", unsafe_allow_html=True)

def user_input():
    with st.sidebar:
        age = st.slider("Age", 20, 80, 45)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
        chol = st.slider("Cholestoral", 100, 400, 200)
        fbs = st.radio("Fasting Blood Sugar", [0, 1], horizontal=True)
        restecg = st.selectbox("Resting ECG", [0, 1, 2])
        thalach = st.slider("Max Heart Rate", 70, 210, 150)
        exang = st.radio("Exercise Angina", [0, 1], horizontal=True)
        oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("Major Vessels", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

    return pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }, index=[0])

input_df = user_input()

# --- Visualizations ---
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📊 Gender vs Health")
    # Purple/Pink Theme Histogram
    fig_sex = px.histogram(df, x="sex", color="target", barmode='group',
                           color_discrete_sequence=['#ce93d8', '#f06292'],
                           labels={'sex': 'Gender (0=F, 1=M)', 'target': 'Heart Disease'})
    fig_sex.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                          font_color="#f8bbd0", showlegend=False)
    st.plotly_chart(fig_sex, use_container_width=True)

with col2:
    st.markdown("### 📈 Heart Rate Flow")
    fig_chol = px.violin(df, y="thalach", x="target", color="target", box=True,
                        color_discrete_sequence=['#ba68c8', '#f48fb1'])
    fig_chol.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#f8bbd0")
    st.plotly_chart(fig_chol, use_container_width=True)

# --- Model ---
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# --- Prediction Action ---
st.markdown("<br>", unsafe_allow_html=True)
if st.button("✨ START AI DIAGNOSIS ✨"):
    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df)
    
    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 1.2])
    
    with res_col1:
        if prediction[0] == 1:
            st.markdown("<div style='background: rgba(255,0,0,0.2); padding: 20px; border-radius: 15px; border-left: 10px solid #ff1744;'>"
                        "<h2 style='color: #ff5252;'>🚨 High Risk Alert</h2>"
                        "<p>Please consult a specialist immediately.</p></div>", unsafe_allow_html=True)
            st.metric("Heart Stress Level", f"{prob[0][1]*100:.1f}%")
        else:
            st.markdown("<div style='background: rgba(0,255,0,0.1); padding: 20px; border-radius: 15px; border-left: 10px solid #00e676;'>"
                        "<h2 style='color: #69f0ae;'>💖 Heart is Safe</h2>"
                        "<p>Your heart profile looks great!</p></div>", unsafe_allow_html=True)
            st.metric("Health Score", f"{prob[0][0]*100:.1f}%")
            
    with res_col2:
        # Beautiful Gauge for Risk
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob[0][1]*100,
            gauge = {
                'axis': {'range': [None, 100], 'tickcolor': "#f48fb1"},
                'bar': {'color': "#f06292"},
                'bgcolor': "rgba(0,0,0,0.3)",
                'steps': [
                    {'range': [0, 40], 'color': "#4a148c"},
                    {'range': [40, 70], 'color': "#880e4f"},
                    {'range': [70, 100], 'color': "#b71c1c"}],
            }
        ))
        fig_gauge.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', font={'color': "#f48fb1", 'family': "Arial"})
        st.plotly_chart(fig_gauge, use_container_width=True)

# --- Stylish Footer Data ---
with st.expander("👑 Deep Dive: Professional Data Insights"):
    st.dataframe(df.style.background_gradient(cmap='PuRd'))