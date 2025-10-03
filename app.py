import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from keras.losses import MeanSquaredError
from datetime import datetime

# -----------------------------
# Load Models & Scalers
# -----------------------------
lstm_model = load_model("models/lstm_model.h5", custom_objects={'mse': MeanSquaredError()})
scaler_X = joblib.load("models/scaler_X.pkl")
scaler_y = joblib.load("models/scaler_y.pkl")
prima_model = joblib.load("models/prima_model.pkl")
prima_scaler = joblib.load("models/prima_scaler.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="📈 Stock Price Predictor | PRIMA & LSTM",
    page_icon="📊",
    layout="centered"
)

# -----------------------------
# App Header
# -----------------------------
st.title("📈 Stock Price Prediction Dashboard")
st.markdown("""
### 👋 Welcome to our **Stock Price Prediction Tool**  

Predict the **Closing Price** of a stock using either:  
- **PRIMA Model (Linear Regression)**  
- **LSTM Deep Learning Network**  

🔍 **Features:**  
- Takes into account real stock market data (Open, High, Low, Volume)  
- Interactive input for stock features  
- Fast prediction with PRIMA or advanced sequential learning with LSTM  
""")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("📊 Input Stock Features")
company_name = st.sidebar.text_input("🏢 Company Name", "Accor")
date = st.sidebar.date_input("📅 Date", datetime.today())
open_price = st.sidebar.number_input("💹 Open Price", min_value=0.0, step=0.01)
daily_high = st.sidebar.number_input("📈 Daily High", min_value=0.0, step=0.01)
daily_low = st.sidebar.number_input("📉 Daily Low", min_value=0.0, step=0.01)
volume = st.sidebar.number_input("📊 Volume", min_value=0.0, step=1.0)
model_choice = st.sidebar.selectbox("🧠 Select Model", ["PRIMA (Linear Regression)", "LSTM Deep Learning"])

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("🚀 Predict Closing Price"):
    with st.spinner("🤖 Calculating prediction..."):
        input_data = np.array([[open_price, daily_high, daily_low, volume]])

        if model_choice == "PRIMA (Linear Regression)":
            input_scaled = prima_scaler.transform(input_data)
            pred = prima_model.predict(input_scaled)
            predicted_price = pred[0]
            st.success(f"💹 Predicted Closing Price (PRIMA) for **{company_name}** on {date}: **${predicted_price:.2f}**")
            st.info("PRIMA Model: Fast traditional regression approach for structured data.")

        else:
            input_scaled = scaler_X.transform(input_data)
            input_scaled = input_scaled.reshape((1, 1, input_scaled.shape[1]))
            pred_scaled = lstm_model.predict(input_scaled, verbose=0)
            predicted_price = scaler_y.inverse_transform(pred_scaled)[0][0]
            st.success(f"💹 Predicted Closing Price (LSTM) for **{company_name}** on {date}: **${predicted_price:.2f}**")
            st.info("LSTM Model: Deep learning approach capturing sequential trends and patterns.")

        st.balloons()

# -----------------------------
# Developer Info
# -----------------------------
st.markdown("---")
st.markdown("""
### 👩‍💻 Developer Information
- **Name:** Md. Abu Rayhan Imran  
- **Role:** AI/ML Engineer | Data Science Researcher  
- **Specialization:** Machine Learning, LSTM, Predictive Modeling  
- **Project:** 📊 DataSynthis_Job_task – Stock Price Forecasting  
- 🌐 [GitHub Repository](https://github.com/yourusername/stock-price-predictor)  
- 📫 Contact: aburayhan2550@gmail.com
""")

st.caption("© 2025 PRIMA & LSTM Stock Predictor | Built with ❤️ using Streamlit & TensorFlow")
