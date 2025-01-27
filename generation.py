import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model

# Load and process data
start_date = "2010-01-01"
end_date = "2020-01-01"
stock_symbol = "AAPL"

# Download stock data
data = yf.download(stock_symbol, start=start_date, end=end_date)
data = data[['Close']]
data.dropna(inplace=True)

# Prepare training and testing datasets
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

# Prepare input-output pairs
def create_dataset(data, lookback=100):
    x, y = [], []
    for i in range(lookback, len(data)):
        x.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

lookback = 100
x_train, y_train = create_dataset(scaled_train_data, lookback)
x_test, y_test = create_dataset(scaled_test_data, lookback)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1)

# Make predictions
predicted_train = model.predict(x_train)
predicted_test = model.predict(x_test)

# Reverse scaling
predicted_train = scaler.inverse_transform(predicted_train.reshape(-1, 1))
predicted_test = scaler.inverse_transform(predicted_test.reshape(-1, 1))
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Save the model
model.save("keras_model.h5")
