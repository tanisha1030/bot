import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load data and model
@st.cache_resource
def load_model_and_data():
    df = pd.read_csv("synthetic_robot_logs.csv")
    model = load_model("botnet_detection_model.h5")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[['packet_size', 'interval']])
    return df, model, scaler, X_scaled

df, model, scaler, X_scaled = load_model_and_data()

st.set_page_config(page_title="Botnet Detection", layout="wide")
st.title("🤖 Botnet Detection in Robotic Network Logs")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# User input for prediction
st.subheader("🔍 Predict a Single Entry")
packet_size = st.number_input("Packet Size", min_value=0.0)
interval = st.number_input("Interval", min_value=0.0)

if st.button("Predict"):
    input_data = np.array([[packet_size, interval]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0][0]
    label = "🔴 Botnet Activity Detected" if prediction > 0.5 else "🟢 Normal Activity"
    st.markdown(f"### Prediction: {label} (Confidence: {prediction:.2f})")

# Optional: Display data distribution
st.subheader("📊 Packet Size vs Interval Scatter Plot")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="packet_size", y="interval", hue="is_botnet", palette="coolwarm", ax=ax)
st.pyplot(fig)
