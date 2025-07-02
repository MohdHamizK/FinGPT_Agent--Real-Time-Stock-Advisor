# === 1. IMPORT LIBRARIES ===
import pandas as pd
import numpy as np
import datetime
import os

# Optional Imports with Graceful Degradation
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
    days = st.sidebar.slider("Number of past days:", min_value=30, max_value=365, value=90)
else:
    print("Streamlit is not available. Please install it using 'pip install streamlit' to run the interactive dashboard.")
    ticker = "AAPL"
    days = min(days, 90)

# === 3. FETCH STOCK DATA ===
end = datetime.datetime.today()
start = end - datetime.timedelta(days=days)

data_path = f"data/{ticker}.xlsx"
if os.path.exists(data_path):
    data = pd.read_excel(data_path, parse_dates=["Date"])
    data = data.set_index("Date")
    data = data.sort_index()
    data = data[(data.index >= start) & (data.index <= end)]

    if st:
        st.code(data.head().to_string())
        st.write("Raw Columns:", data.columns)
else:
    if st:
        st.warning(f"No local Excel file found for {ticker}.xlsx in /data.")
        st.stop()
    else:
        print(f"Missing file: data/{ticker}.xlsx")
        exit()

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
        api_key= "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",
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