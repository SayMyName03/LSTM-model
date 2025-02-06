import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import streamlit as st

# Define the time period
start = '2010-01-10'
end = '2025-2-2'

# Streamlit App
st.set_page_config(page_title="StockSage", layout="wide")

# Landing Page with Video Background
st.markdown("""
    <style>
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
        }
        .video-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
        }
        .content {
            position: relative;
            z-index: 1;
            text-align: center;
            color: white;
        }
    </style>
    <div class="video-container">
        <video autoplay loop muted style="width: 100%; height: 100%; object-fit: cover;">
            <source src="C:\Users\saymy\LSTM-model\Background\204306-923909642_small.mp4" type="video/mp4">
        </video>
    </div>
    <div class="content">
        <h1>Welcome to StockSage</h1>
        <h3>Your AI-powered stock analysis tool</h3>
    </div>
""", unsafe_allow_html=True)

# Stock Ticker Input
user_input = st.text_input('Enter Stock Ticker', '')
submit_button = st.button("Analyze Stock")

if submit_button and user_input:
    # Download stock data
    df = yf.download(user_input, start=start, end=end)
    
    if df.empty:
        st.error("No data found for the ticker. Please check the ticker symbol.")
    else:
        # Display Stock Details
        st.markdown(f"<h1 style='text-align: center;'>{user_input} Stock Analysis</h1>", unsafe_allow_html=True)
        
        # Stock Metrics
        st.subheader("Stock Metrics")
        st.write(df.describe())

        # Raw Data
        st.subheader("Raw Data")
        st.write(df)

        # Closing Price vs Time Chart
        st.subheader('Closing Price vs Time Chart')
        fig = plt.figure(figsize=(12,6))
        plt.plot(df.Close)
        st.pyplot(fig)

        # Closing Price with 100MA
        st.subheader('Closing Price vs Time with 100-Day Moving Average')
        ma100 = df.Close.rolling(100).mean()
        fig = plt.figure(figsize=(12,6))
        plt.plot(ma100, label="100-Day MA")
        plt.plot(df.Close, label="Closing Price")
        plt.legend()
        st.pyplot(fig)

        # Closing Price with 100MA & 200MA
        st.subheader('Closing Price vs Time with 100-Day & 200-Day Moving Averages')
        ma200 = df.Close.rolling(200).mean()
        fig = plt.figure(figsize=(12,6))
        plt.plot(ma100, 'r', label="100-Day MA")
        plt.plot(ma200, 'g', label="200-Day MA")
        plt.plot(df.Close, 'b', label="Closing Price")
        plt.legend()
        st.pyplot(fig)

        # Splitting Data
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

        scaler = MinMaxScaler(feature_range=(0,1))
        data_training_array = scaler.fit_transform(data_training)

        # Load Pre-trained Model
        model = load_model('keras_model.h5')

        # Prepare Test Data
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.transform(final_df)

        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100: i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        y_predicted = model.predict(x_test)

        # Inverse Transform Predictions
        y_predicted = scaler.inverse_transform(y_predicted.reshape(-1, 1))
        y_test = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))

        # Prediction vs Original
        st.subheader('Prediction vs Original')
        fig2 = plt.figure(figsize=(12,6))
        plt.plot(y_test, 'b', label='Original Price')
        plt.plot(y_predicted, 'r', label='Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig2)

        # Function to Predict Future Prices
        def predict_future(stock_model, scaler, last_100_data, future_days=30):
            future_input = last_100_data.copy()
            predicted_prices = []

            for _ in range(future_days):
                input_data = future_input[-100:].reshape(1, 100, 1)
                predicted_price = stock_model.predict(input_data)[0, 0]
                predicted_prices.append(predicted_price)
                future_input = np.append(future_input, predicted_price)

            predicted_prices = np.array(predicted_prices).reshape(-1, 1)
            return scaler.inverse_transform(predicted_prices)

        # Predict Future Prices
        last_100_data = data_training_array[-100:]
        future_predictions = predict_future(model, scaler, last_100_data, future_days=30)

        st.subheader('Future Price Predictions')
        fig3 = plt.figure(figsize=(12,6))
        plt.plot(future_predictions, 'g', label='Future Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig3)
