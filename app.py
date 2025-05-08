import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="LSTM Stock Predictor", layout="centered")
st.title("📈 LSTM Stock Price Predictor")

ticker = st.text_input("Enter Stock Ticker:", "AAPL")

if st.button("Predict"):
    try:
        data = yf.Ticker(ticker).history(period="10y")
        data = data[['Close']].dropna()
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data.values)

        train_size = int(len(scaled) * 0.8)
        train = scaled[:train_size]
        test = scaled[train_size - 60:]

        def create_dataset(ds, lookback=60):
            X, y = [], []
            for i in range(lookback, len(ds)):
                X.append(ds[i - lookback:i, 0])
                y.append(ds[i, 0])
            return np.array(X), np.array(y)

        lookback = 60
        X_train, y_train = create_dataset(train, lookback)
        X_test, y_test = create_dataset(test, lookback)
        X_train = X_train.reshape((X_train.shape[0], lookback, 1))
        X_test = X_test.reshape((X_test.shape[0], lookback, 1))

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        st.subheader("📊 Actual vs Predicted Prices")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(data.index[train_size:], actual, label="Actual")
        ax.plot(data.index[train_size:], predictions, label="Predicted")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.set_title(f"{ticker} Stock Prediction")
        ax.legend()
        st.pyplot(fig)

        def predict_future(model, last_seq, days):
            preds = []
            seq = last_seq.copy()
            for _ in range(days):
                pred = model.predict(seq.reshape(1, lookback, 1))[0, 0]
                preds.append(pred)
                seq = np.append(seq[1:], pred).reshape(lookback, 1)
            return scaler.inverse_transform(np.array(preds).reshape(-1, 1))

        st.subheader("🔮 Future Predictions")
        last_seq = X_test[-1]
        horizons = {
            "1 Day": 1,
            "1 Week": 5,
            "1 Month": 22,
            "6 Months": 126,
            "1 Year": 252
        }

        for label, days in horizons.items():
            price = predict_future(model, last_seq, days)[-1][0]
            st.write(f"**{label}:** ${price:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
