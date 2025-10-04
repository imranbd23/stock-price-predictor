# 📈 Stock Price Prediction Web App (PRIMA & LSTM)

A machine learning web application that predicts the **Closing Price** of a stock based on historical features using either a **PRIMA (Linear Regression) Model** or a **Deep Learning LSTM Model**. Built with **Streamlit**, **TensorFlow**, **scikit-learn**, and **joblib**.
💡 Check it out here: https://lnkd.in/g3NJpqvA
📝 Blog Post: https://surl.li/eqyykf
---

## 🚀 Features
- Predict stock **Closing Price** from user inputs: Open, High, Low, Volume
- Choose between:
  - **PRIMA Model (Linear Regression)** – traditional machine learning
  - **LSTM Model** – deep learning sequential prediction
- Real-time interactive **web interface** with Streamlit
- Easy to deploy on **Streamlit Cloud**, **Hugging Face**, or **Heroku**
- Scalable and ready for future enhancements

---

## 📂 Project Structure

stock-price-predictor/
├─ app.py # Streamlit web application
├─ models/
│ ├─ lstm_model.h5 # Trained LSTM model
│ ├─ scaler_X.pkl # Feature scaler for LSTM
│ ├─ scaler_y.pkl # Target scaler for LSTM
│ ├─ prima_model.pkl # Trained PRIMA Linear Regression model
│ └─ prima_scaler.pkl # Feature scaler for PRIMA
├─ requirements.txt # Python dependencies
└─ README.md # Project documentation
---

## 🛠️ Installation & Run Locally

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/stock-price-predictor.git
cd stock-price-predictor

## Install dependencies

pip install -r requirements.txt

## Run this 
streamlit run app.py
Open your browser → App will be available at http://localhost:8501

🧠 How It Works

PRIMA Model: Fast linear regression approach for structured input data.

LSTM Model: Captures sequential patterns in stock price trends for more advanced prediction.

User Inputs: Company Name, Date, Open, High, Low, and Volume.

Prediction Output: Forecasted Closing Price displayed interactively.

👩‍💻 Developer Information

Name: Md. Abu Rayhan Imran

Role: AI/ML Engineer | Data Science Researcher

Specialization: Machine Learning, Deep Learning (LSTM), Predictive Modeling

Project Name: DataSynthis_Job_task – Stock Price Forecasting

GitHub Repository: Link

Contact: your_email@example.com

⚡ Notes

Ensure models/ folder contains all trained models and scalers.

Requires Python ≥3.8 for TensorFlow & Streamlit compatibility.

For longer time-series predictions, LSTM may require more memory and computation time.


I can also **write a full `requirements.txt`** optimized for both PRIMA and LSTM models to make the setup smoother.  

Do you want me to create that next?

