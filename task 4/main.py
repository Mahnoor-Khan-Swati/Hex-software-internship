import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.title("Heart Disease Prediction App ")

df = pd.read_csv(r"C:\Users\user\Downloads\Heart Disease Prediction Model\heart.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Summary")
st.write(df.describe())

st.subheader("Dataset Info")
buffer = StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.subheader("Gender vs Target")
plt.figure(figsize=(8,5))
sns.countplot(x="sex", data=df, hue="target")
plt.title("Gender vs Heart Disease Target")
st.pyplot(plt)

st.subheader("Boxplot of Features")
plt.figure(figsize=(10,6))
sns.boxplot(data=df)
st.pyplot(plt)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.subheader("Model Accuracy on Test Set")
st.write(f"{accuracy*100:.2f}%")

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

st.sidebar.header("Enter Patient Details")

def user_input():
    age = st.sidebar.slider("Age", 20, 80, 40)
    sex = st.sidebar.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.sidebar.slider("Chest Pain Type (0-3)", 0, 3, 1)
    trestbps = st.sidebar.slider("Resting BP", 80, 200, 120)
    chol = st.sidebar.slider("Cholesterol", 100, 400, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar (0/1)", [0, 1])
    restecg = st.sidebar.slider("Rest ECG (0-2)", 0, 2, 1)
    thalach = st.sidebar.slider("Max Heart Rate", 70, 210, 150)
    exang = st.sidebar.selectbox("Exercise Angina (0/1)", [0, 1])
    oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.sidebar.slider("Slope (0-2)", 0, 2, 1)
    ca = st.sidebar.slider("CA (0-4)", 0, 4, 0)
    thal = st.sidebar.slider("Thal (0-3)", 0, 3, 1)

    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input()

if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.error(f"High Risk of Heart Disease! \nProbability: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.success(f"Low Risk of Heart Disease \nProbability: {prediction_proba[0][0]*100:.2f}%")