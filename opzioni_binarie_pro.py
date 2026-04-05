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
st.markdown("**Multi-Asset • Auto Refresh • XGBoost**")

# ====================== SIDEBAR ======================
st.sidebar.header("⚙️ Impostazioni")

assets = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "BTC-USD", "ETH-USD", "AAPL", "TSLA", "NVDA"]
selected_assets = st.sidebar.multiselect("Seleziona Asset", assets, default=["EURUSD=X", "BTC-USD"])

interval = st.sidebar.selectbox("Intervallo candele", ["1m", "5m", "15m"])
period = st.sidebar.selectbox("Periodo storico", ["7d", "30d", "60d"])
target_minutes = st.sidebar.slider("Previsione prossimi minuti", 1, 15, 5)

confidence_threshold = st.sidebar.slider("Soglia minima confidenza", 60, 90, 70)
volume_filter = st.sidebar.checkbox("Filtro Volume alto", value=True)
volume_multiplier = st.sidebar.slider("Moltiplicatore volume", 1.0, 3.0, 1.5)

enable_audio = st.sidebar.checkbox("Audio Alert (>70%)", value=True)

st.sidebar.header("🔄 Refresh Automatico")
refresh_seconds = st.sidebar.selectbox("Aggiorna ogni", [15, 30, 60, 120, 300], index=2)
auto_refresh = st.sidebar.checkbox("Attiva Refresh Automatico", value=True)

# ====================== FUNZIONI ======================
@st.cache_data(ttl=180)
def get_data(ticker, period, interval):
    return yf.download(ticker, period=period, interval=interval, progress=False)

# ====================== ANALISI ======================
all_signals = []

for ticker in selected_assets:
    with st.spinner(f"Analizzando {ticker}..."):
        df = get_data(ticker, period, interval)
        if df.empty or len(df) < 100:
            continue

        data = df.copy()
        data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

        future_price = data["Close"].shift(-target_minutes)
        data["Target"] = (future_price > data["Close"]).astype(int)
        data = data.dropna()

        feature_cols = [col for col in data.columns if col not in ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Target"]]

        X = data[feature_cols]
        y = data["Target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

        # Modello XGBoost
        model = xgb.XGBClassifier(
            n_estimators=400, 
            max_depth=7, 
            learning_rate=0.05, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=42
        )
        model.fit(X_train, y_train)

        latest_features = data[feature_cols].iloc[-1:].values
        prob = model.predict_proba(latest_features)[0]

        confidence = max(prob) * 100
        direction = "🟢 UP" if prob[1] > prob[0] else "🔴 DOWN"
        volume_ok = data["Volume"].iloc[-1] > (data["Volume"].mean() * volume_multiplier)

        if confidence >= confidence_threshold and (not volume_filter or volume_ok):
            all_signals.append({
                "Asset": ticker,
                "Direzione": direction,
                "Confidenza": f"{confidence:.1f}%",
                "Prob UP": f"{prob[1]*100:.1f}%",
                "Prezzo": f"{data['Close'].iloc[-1]:.4f}",
                "Volume OK": "✅" if volume_ok else "❌"
            })

            if enable_audio and confidence >= 70:
                st.audio("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3", autoplay=True)

# ====================== RISULTATI ======================
st.subheader("🔴 Segnali Forti Attivi")

if all_signals:
    st.success(f"✅ **{len(all_signals)} Segnali forti** rilevati!")
    st.dataframe(pd.DataFrame(all_signals), use_container_width=True)
else:
    st.info("⏳ Nessun segnale sopra la soglia in questo momento...")

# Grafici
for ticker in selected_assets[:3]:
    df = get_data(ticker, period, interval)
    if not df.empty:
        fig = go.Figure(data=[go.Candlestick(x=df.index[-200:],
                        open=df['Open'].iloc[-200:], high=df['High'].iloc[-200:],
                        low=df['Low'].iloc[-200:], close=df['Close'].iloc[-200:])])
        fig.update_layout(title=f"{ticker} — Ultime 200 candele", height=450)
        st.plotly_chart(fig, use_container_width=True)

st.caption(f"Ultimo aggiornamento: {datetime.now().strftime('%H:%M:%S')}")

# Auto Refresh
if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()

st.warning("⚠️ Solo scopo educativo - Non è un consiglio finanziario")
