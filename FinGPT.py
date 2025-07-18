# === 1. IMPORT LIBRARIES ===
import pandas as pd
import numpy as np
import datetime
import os

try:
    import streamlit as st
except ImportError:
    st = None 

try:
    import yfinance as yf
except ImportError:
    yf = None

# === 2. STREAMLIT INTERFACE SETUP ===

if st:
    st.set_page_config(page_title="FinGPT Agent", layout="wide")
    st.title("FinGPT Agent: Real-Time Stock Advisor")

    ticker = st.sidebar.selectbox("Choose Stock Ticker:", ["AAPL", "TSLA", "AMZN", "NVDA"])
    days = st.sidebar.slider("Number of past days:", min_value=30, max_value=365, value=150)
else:
    print("Streamlit is not available. Please install it using 'pip install streamlit' to run the interactive dashboard.")
    ticker = ("AAPL", "TSLA", "AMZN", "NVDA")
    days = min(days, 90)

# === 3. FETCH STOCK DATA ===

ticker_file_map = {
    "AAPL": "AAPL.xlsx",
    "TSLA": "TSLA.xlsx",
    "AMZN": "AMZN.xlsx",
    "NVDA": "NVDA.xlsx"
}

data_path = ticker_file_map.get(ticker)

try:
    data = pd.read_excel(data_path, index_col=0, parse_dates=True)
    data.columns = data.columns.str.strip()
except Exception as e:
    if st:
        st.error(f"Error loading data file: {e}")
        st.stop()
    else:
        print(f"Error loading data file: {e}")
        exit()

if data.empty:
    if st:
        st.warning("No data found. Try a different ticker.")
        st.stop()
    else:
        print("No stock data available. Exiting.")
        exit()

if st:
    st.subheader(f"Stock Price for {ticker}")
    st.line_chart(data['Close'])
    st.code(data.head().to_string())
else:
    print(data['Close'].tail())

# === 4. SIMPLE FORECAST ===

data['Return'] = data['Close'].pct_change()
data['Signal'] = np.where(data['Return'] > 0, 1, -1)
latest_price = data['Close'].iloc[-1]
recent_trend = data['Signal'].iloc[-5:].sum()
recent_performance = data['Return'].iloc[-5:].mean()

# === 5. GPT-BASED NEWS SENTIMENT VIA OPENAI ===

try:
    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=st.secrets["OPENAI_APIKEY"]
    )

    def fetch_news_summary(ticker):
        news_text = f"{ticker} stock has shown strong quarterly earnings and increased investor confidence."
        response = client.chat.completions.create(
            model="openai/gpt-4o",
            max_tokens=500,
            messages=[
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": f"Summarize sentiment and impact of this news: {news_text}"}
            ]
        )
        return response.choices[0].message.content

    sentiment_summary = fetch_news_summary(ticker)

    if st:
        st.subheader("News Sentiment Summary")
        st.info(sentiment_summary)
    else:
        print("Sentiment Summary:", sentiment_summary)

except Exception as e:
    if st:
        st.warning(f"Sentiment analysis not available: {str(e)}")
    else:
        print("Sentiment analysis not available:", str(e))

# === 6. AGENT DECISION LOGIC ===

sentiment_positive = "positive" in sentiment_summary.lower()

recommendation = "Hold"
if recent_performance > 0.005 and sentiment_positive:
    recommendation = "Buy"
elif recent_trend < -2:
    recommendation = "Sell"

if st:
    st.subheader("FinGPT Agent Recommendation")
    st.success(f"Recommendation: {recommendation}")
else:
    print("Recommendation:", recommendation)

# === 7. SHOW RAW DATA ===
if st:
    with st.expander("Show Raw Data"):
        st.dataframe(data.tail())
else:
    print("Raw Data Preview:")
    print(data.tail())
