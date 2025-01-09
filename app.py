import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import streamlit as st

# Set the title of the app
st.title("Apple Stock Price Prediction")

# Define the time period
start = '2010-01-10'
end = '2019-12-31'

# Download the data
df = yf.download('AAPL', start=start, end=end)
df = df.reset_index()

# Drop unnecessary columns
df = df.drop(['Date', 'Adj Close'], axis=1)

# Plot the closing prices
st.subheader('Closing Prices')
st.line_chart(df['Close'])

# Calculate moving averages
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()

# Plot moving averages
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Close Price')
plt.plot(ma100, 'r', label='100-Day MA')
plt.plot(ma200, 'g', label='200-Day MA')
plt.title('AAPL Closing Prices with Moving Averages')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
st.pyplot(plt)

# Splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Prepare training data
x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping data for LSTM
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

# Building the LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Show the model summary in the app
st.subheader('LSTM Model Summary')
st.text(model.summary())
