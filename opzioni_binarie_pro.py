import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from ta import add_all_ta_features
import time

st.set_page_config(page_title="Opzioni Binarie PRO", layout="wide")
st.title("🎯 Previsioni Opzioni Binarie - PRO")

assets = ["EURUSD=X", "BTC-USD", "ETH-USD", "AAPL"]
selected_assets = st.sidebar.multiselect("Asset", assets, default=["EURUSD=X", "BTC-USD"])

interval = st.sidebar.selectbox("Intervallo", ["1m", "5m"])
period = st.sidebar.selectbox("Periodo", ["7d", "30d"])
target_minutes = st.sidebar.slider("Minuti previsione", 1, 15, 5)

confidence_threshold = st.sidebar.slider("Confidenza minima %", 60, 90, 70)

@st.cache_data(ttl=60)
def get_data(ticker):
    return yf.download(ticker, period=period, interval=interval, progress=False)

all_signals = []

for ticker in selected_assets:
    df = get_data(ticker)
    if len(df) < 50: continue
        
    data = df.copy()
    data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    
    future = data["Close"].shift(-target_minutes)
    data["Target"] = (future > data["Close"]).astype(int)
    data = data.dropna()
    
    feature_cols = [col for col in data.columns if col not in ["Open","High","Low","Close","Adj Close","Volume","Target"]]
    
    X = data[feature_cols]
    y = data["Target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    latest = data[feature_cols].iloc[-1:].values
    prob = model.predict_proba(latest)[0]
    confidence = max(prob) * 100
    direction = "🟢 UP" if prob[1] > 0.5 else "🔴 DOWN"
    
    if confidence >= confidence_threshold:
        all_signals.append([ticker, direction, f"{confidence:.1f}%", f"{data['Close'].iloc[-1]:.4f}"])

st.subheader("Segnali Attivi")
if all_signals:
    st.success(f"Trovati {len(all_signals)} segnali!")
    st.table(pd.DataFrame(all_signals, columns=["Asset", "Direzione", "Confidenza", "Prezzo"]))
else:
    st.info("Nessun segnale forte al momento")

st.caption(f"Aggiornato: {datetime.now().strftime('%H:%M:%S')}")
