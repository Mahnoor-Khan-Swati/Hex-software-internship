import streamlit as st
import joblib

# 1. Page Config
st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="💛✨",
    layout="centered"
)

# 2. Professional Yellow & Gold Theme (CSS)
st.markdown("""
    <style>
    /* Animated Dark & Gold Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #1a1a1a, #332b00, #4d4100, #1a1a1a);
        background-size: 400% 400%;
        animation: gradient 12s ease infinite;
        color: white;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glassmorphism Effect for Main Container */
    .main-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 215, 0, 0.3); /* Gold Border */
        box-shadow: 0 8px 32px 0 rgba(255, 215, 0, 0.2);
    }

    /* Shining Gold Title */
    .main-title {
        font-size: 45px;
        font-weight: bold;
        text-align: center;
        color: #ffcc00; /* Professional Gold */
        text-shadow: 0 0 10px #ffcc00, 0 0 20px #997a00;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #fff176;
        font-size: 18px;
        margin-bottom: 20px;
    }

    /* Golden Glowing Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #ffcc00, #997a00);
        color: #1a1a1a !important; /* Dark text on yellow button looks professional */
        border: none;
        border-radius: 12px;
        height: 3.5em;
        width: 100%;
        font-size: 17px;
        font-weight: bold;
        transition: 0.3s;
        box-shadow: 0 4px 15px rgba(255, 204, 0, 0.4);
    }

    .stButton>button:hover {
        transform: scale(1.03);
        box-shadow: 0 0 25px #ffcc00;
        color: black !important;
    }

    /* Sidebar Styling (Dark Grey Glass) */
    [data-testid="stSidebar"] {
        background: rgba(20, 20, 20, 0.9);
        border-right: 1px solid rgba(255, 215, 0, 0.2);
    }
    
    /* Text Input Area (Professional Light Yellow bg) */
    .stTextArea textarea {
        background-color: #fffde7 !important; 
        color: #332b00 !important; 
        font-size: 18px !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        border: 2px solid #ffcc00 !important;
    }

    /* Professional Yellow Footer */
    .footer-text {
        color: #ffcc00 !important; 
        text-align: center;
        font-size: 14px;
        font-weight: 500;
        letter-spacing: 1px;
        padding-top: 25px;
    }

    /* Label text color */
    label, p, h3 {
        color: #ffcc00 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load Model & Vectorizer
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except:
    pass 

# 4. Header Section
st.markdown('<div class="main-title"> Sentiment Analysis </div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Professional Golden NLP Analyzer</div>', unsafe_allow_html=True)

# Main Container Start
st.markdown('<div class="main-card">', unsafe_allow_html=True)

# 5. Sidebar
st.sidebar.markdown("<h2 style='color:#ffcc00;'>⚙️ Settings</h2>", unsafe_allow_html=True)
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Product Review", "Service Review", "General Text"]
)
confidence_display = st.sidebar.checkbox("Show Prediction Confidence")
st.sidebar.markdown("---")
st.sidebar.info("Developed by Mahnoor Khan 💛")

# 6. Main Input Section
st.subheader("✍️ Enter Your Review")
text = st.text_area("Type your review here...", height=150, placeholder="Share your thoughts in gold...")

col1, col2 = st.columns(2)
with col1:
    predict_button = st.button("🔍 Predict Sentiment")
with col2:
    clear_button = st.button("🗑 Clear")

if clear_button:
    st.rerun()

# 7. Prediction Logic
if predict_button:
    if text.strip() != "":
        try:
            transformed_text = vectorizer.transform([text])
            prediction = model.predict(transformed_text)[0]

            st.markdown("<br>", unsafe_allow_html=True)

            if prediction == "positive":
                st.success("😊 Positive Sentiment Detected!")
                st.balloons()
            elif prediction == "negative":
                st.error("😠 Negative Sentiment Detected!")
            else:
                st.warning("😐 Neutral Sentiment Detected!")

            if confidence_display:
                probabilities = model.predict_proba(transformed_text)
                confidence = max(probabilities[0]) * 100
                st.info(f"📊 Confidence Score: {confidence:.2f}%")
        except:
            st.warning("UI looks premium! (Model files needed for actual prediction)")
    else:
        st.warning("⚠️ Please enter review text.")

st.markdown('</div>', unsafe_allow_html=True) # Main Container End

# 8. Footer Section
st.markdown('<p class="footer-text">© 2026 Sentiment Analysis NLP Project | HexaSoftware Internship</p>', unsafe_allow_html=True)