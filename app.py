import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# --- Load the model ---s

# Replace the path below with your actual saved model file (.keras or .h5)
model_path = r"C:\Users\sid24\OneDrive\Desktop\Machine learning\Future_Bitcoin_Prediction\Bitcoin_Price_prediction_Model.keras"

model = load_model(model_path)

# --- Streamlit UI ---

st.header('Bitcoin Price Prediction Model')
st.subheader('Bitcoin Price Data')

# Download historical Bitcoin data
data = yf.download('BTC-USD', start='2015-01-01', end='2023-11-30')
data = data.reset_index()

st.write(data)

# Line chart of Bitcoin Close Price
st.subheader('Bitcoin Closing Price Line Chart')
st.line_chart(data['Close'])

# Prepare data for prediction

# Use only the 'Close' column for training/testing
close_data = data[['Close']]

# Split data into train and test sets
train_data = close_data[:-100]
test_data = close_data[-200:]  # make sure test_data length > base_days

# Scale data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

base_days = 100  # number of days to look back for predictions

# Prepare test inputs (x) and outputs (y)
x_test = []
y_test = []

for i in range(base_days, len(test_scaled)):
    x_test.append(test_scaled[i - base_days:i, 0])
    y_test.append(test_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Reshape input to 3D (samples, time steps, features)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict on test data
st.subheader('Predicted vs Original Prices')
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
originals = scaler.inverse_transform(y_test.reshape(-1, 1))

# Show comparison in a dataframe
comparison_df = pd.DataFrame({
    'Predicted Price': predictions.flatten(),
    'Original Price': originals.flatten()
})

st.write(comparison_df)

# Plot predicted vs original
st.subheader('Predicted vs Original Prices Chart')
st.line_chart(comparison_df)

# --- Predict future prices ---

st.subheader('Predicted Future Bitcoin Prices')

# Start with the last 'base_days' from the test data
future_input = test_scaled[-base_days:, 0]

future_predictions = []

for _ in range(5):  # predict next 5 days
    # Prepare input shape (1, base_days, 1)
    input_reshaped = future_input.reshape(1, base_days, 1)

    # Predict next price
    next_pred = model.predict(input_reshaped)[0, 0]

    # Append prediction to the future_predictions list
    future_predictions.append(next_pred)

    # Update input sequence by appending prediction and removing first element
    future_input = np.append(future_input[1:], next_pred)

# Inverse transform future predictions to original scale
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Display future predictions as a dataframe
future_df = pd.DataFrame(future_predictions, columns=['Predicted Price'])

st.write(future_df)
st.line_chart(future_df)
