import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model
import streamlit as st

model_path = os.path.join(os.getcwd(), 'keras_model.h5')


# Define the time period
start = '2010-01-10'
end = '2019-12-31'

# Set the title of the app
st.title("StockSage")

# Get user input for stock ticker
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Download the data using user input
df = yf.download(user_input, start=start, end=end)

# Check if the DataFrame is empty
if df.empty:
    st.error("No data found for the ticker. Please check the ticker symbol.")
else:
    # Describing Data 
    st.subheader('Data from 2010 - 2019')
    st.write(df.describe())

    # Optionally, display the DataFrame as well
    st.subheader('Raw Data')
    st.write(df)

# Visualizations
st.subheader('Closing price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)

# Splitting data into training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)



# Load my model
model = load_model(model_path)

# Testing part

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


# Final graph

st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)