import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime
import matplotlib.pyplot as plt
import sqlite3
from statsmodels.tsa.arima.model import ARIMA
import random

# Title and Description
st.title("Cryptocurrency Analysis App")
st.write("An interactive app for fetching, analyzing, and visualizing cryptocurrency data using the CoinGecko API.")

# Sidebar for inputs
st.sidebar.header("Settings")
currency = st.sidebar.selectbox("Select Currency", options=["usd", "eur", "gbp"], index=0)
alert_threshold = st.sidebar.slider("Alert Threshold (% Change)", min_value=1, max_value=50, value=10)
show_forecast = st.sidebar.checkbox("Show 7-day Forecast")

# Fetch Data
@st.cache_data
def fetch_crypto_data(currency):
    url = 'https://api.coingecko.com/api/v3/coins/markets'
    params = {
        'vs_currency': currency,
        'order': 'market_cap_desc',
        'per_page': 100,
        'page': 1,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        st.error(f"Error fetching data: {response.status_code}")
        return None

df = fetch_crypto_data(currency)

if df is not None:
    # Process data
    df = df[['id', 'current_price', 'market_cap', 'price_change_percentage_24h', 'ath', 'atl']]
    today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df['timestamp'] = today

    # Save data
    if not os.path.exists("crypto_data"):
        os.makedirs("crypto_data")
    df.to_csv(f'crypto_data/daily_data_{today}.csv', index=False)

    # Add 7-day moving average
    df['7_day_MA'] = df['current_price'].rolling(window=7).mean()

    # RSI calculation
    def calculate_rsi(data, window=14):
        delta = data['current_price'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['RSI'] = calculate_rsi(df)

    # Alerts for significant price changes
    significant_changes = df[abs(df['price_change_percentage_24h']) >= alert_threshold]
    if not significant_changes.empty:
        st.write("### Significant Price Changes")
        for _, row in significant_changes.iterrows():
            if row['price_change_percentage_24h'] > 0:
                st.write(f"🔼 {row['id']} has increased by {row['price_change_percentage_24h']:.2f}%")
            else:
                st.write(f"🔻 {row['id']} has decreased by {row['price_change_percentage_24h']:.2f}%")

    # Sentiment analysis (placeholder)
    def analyze_sentiment(crypto_id):
        return random.choice(['Positive', 'Negative', 'Neutral'])

    df['sentiment'] = df['id'].apply(analyze_sentiment)

    # Visualization: Top 10 by Market Cap
    st.write("### Top 10 Cryptocurrencies by Market Cap")
    top_10 = df.nlargest(10, 'market_cap')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(top_10['id'], top_10['market_cap'], color='blue')
    ax.set_xlabel("Cryptocurrency")
    ax.set_ylabel("Market Cap (USD)")
    ax.set_title("Top 10 Cryptocurrencies by Market Cap")
    ax.set_xticklabels(top_10['id'], rotation=45)
    st.pyplot(fig)

    # Save data to SQLite
    conn = sqlite3.connect("crypto_data.db")
    df.to_sql("crypto_data", conn, if_exists="append", index=False)
    conn.close()

    # ARIMA forecast
    if show_forecast:
        st.write("### 7-Day Price Forecast (ARIMA)")
        historical_prices = df['current_price'].dropna().values
        model = ARIMA(historical_prices, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=7)
        st.write(f"Forecast: {forecast}")

    # Sharpe Ratio calculation
    risk_free_rate = 0.01
    returns = df['price_change_percentage_24h'] / 100
    sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std()
    st.write(f"### Sharpe Ratio: {sharpe_ratio:.2f}")
else:
    st.error("Failed to load data.")
