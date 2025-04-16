import streamlit as st
import matplotlib.pyplot as plt
from data_loader import load_data
from model import prepare_data, build_and_predict_model

st.set_page_config(page_title="Stock Predictor", layout="centered")

st.title("📈 Stock Price Predictor")
ticker = st.text_input("Enter stock ticker (e.g., AAPL)", "AAPL")

if st.button("Train and Predict"):
    df = load_data(ticker)

    if df is None or df.empty:
        st.error("❌ Failed to load data. Check the ticker.")
    else:
        X, y, scaler = prepare_data(df)
        model, prediction = build_and_predict_model(X, y, scaler)

        st.success("✅ Prediction Complete")
        st.write(f"📍 **Next Day Prediction: {prediction:.2f}**")

        # 📊 Price Chart
        st.subheader("📊 Historical Prices (High + Low)")
        fig, ax = plt.subplots()
        ax.plot(df.index, df["High"], label="High")
        ax.plot(df.index, df["Low"], label="Low")
        ax.legend()
        st.pyplot(fig)

        # 📈 Indicators
        st.subheader("📈 Indicators (SMA & RSI)")
        fig, ax = plt.subplots()
        ax.plot(df.index, df["SMA_20"], label="SMA 20", color="orange")
        ax.plot(df.index, df["SMA_50"], label="SMA 50", color="magenta")
        ax.plot(df.index, df["RSI"], label="RSI", color="green")
        ax.legend()
        st.pyplot(fig)
