"""
Opzioni Binarie - Analisi Tecnica Avanzata
Versione Finale Robusta (2026)
- Supporto CSV + yfinance
- Calcolo sicuro indicatori (evita errori 1-dimensional)
- Segnali di trading per Call/Put
- Grafico interattivo con Plotly
"""

import pandas as pd
import pandas_ta as ta
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

# ========================= CONFIGURAZIONE =========================
DATA_SOURCE = "csv"          # "csv" oppure "yfinance"

# CSV
CSV_PATH = Path("data.csv")  # <-- CAMBIA con il tuo file

# yfinance
TICKER = "AAPL"              # Es: "EURUSD=X", "BTC-USD", "SPY"
PERIOD = "1y"
INTERVAL = "1d"

# Parametri segnali
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
ADI_THRESHOLD = 0.0          # soglia per ADI (puoi adattare)

# ================================================================

def load_data() -> pd.DataFrame:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Caricamento dati...")
    
    if DATA_SOURCE == "yfinance":
        df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)
        df = df.reset_index()
        if 'Date' in df.columns:
            df = df.rename(columns={'Date': 'date'})
    elif DATA_SOURCE == "csv":
        if not CSV_PATH.exists():
            raise FileNotFoundError(f"File non trovato: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
    else:
        raise ValueError("DATA_SOURCE deve essere 'csv' o 'yfinance'")

    # Standardizzazione colonne
    df = df.rename(columns=str.lower)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

    df = df.sort_index()
    required = ['high', 'low', 'close', 'volume']
    df = df.dropna(subset=required)
    df = df[~df.index.duplicated(keep='first')]

    print(f"✅ Dati pronti: {len(df)} righe")
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Calcolo indicatori...")

    df = df.copy()

    # Forza Series 1D (previene errori pandas_ta)
    for col in ['high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.Series(df[col].values, index=df.index)

    # Indicatori
    df['ADI'] = ta.ad(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['SMA_20'] = ta.sma(df['close'], length=20)
    df['EMA_9'] = ta.ema(df['close'], length=9)
    macd = ta.macd(df['close'])
    df['MACD'] = macd['MACD_12_26_9']
    bb = ta.bbands(df['close'], length=20)
    df['BB_upper'] = bb['BBU_20_2.0']
    df['BB_lower'] = bb['BBL_20_2.0']

    print("✅ Indicatori calcolati correttamente")
    return df


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    print("Generazione segnali di trading...")

    df = df.copy()
    df['Signal'] = 'Hold'

    # Segnali semplici per opzioni binarie
    call_condition = (
        (df['RSI'] < RSI_OVERSOLD) & 
        (df['close'] > df['BB_lower']) & 
        (df['ADI'] > ADI_THRESHOLD)
    )

    put_condition = (
        (df['RSI'] > RSI_OVERBOUGHT) & 
        (df['close'] < df['BB_upper']) & 
        (df['ADI'] < ADI_THRESHOLD)
    )

    df.loc[call_condition, 'Signal'] = 'Call'
    df.loc[put_condition, 'Signal'] = 'Put'

    # Segnale extra: crossover EMA9 sopra SMA20 → Call
    df['Signal'] = df['Signal'].where(
        df['EMA_9'] > df['SMA_20'], 
        other=df['Signal'].where(df['Signal'] != 'Hold', 'Call')
    ).where(
        df['EMA_9'] <= df['SMA_20'], 
        other=df['Signal']
    )

    print(f"✅ Segnali generati - Call: { (df['Signal']=='Call').sum() } | Put: { (df['Signal']=='Put').sum() }")
    return df


def create_chart(df: pd.DataFrame, ticker: str = "Asset"):
    print("Creazione grafico interattivo...")
    
    fig = make_subplots(rows=3, cols=1, 
                        shared_xaxes=True,
                        row_heights=[0.5, 0.2, 0.3],
                        subplot_titles=("Prezzo e Bollinger", "RSI", "ADI"))

    # Candlestick + Bollinger
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close'],
                                 name="OHLC"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name="BB Upper", line=dict(color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name="BB Lower", line=dict(color='green', dash='dash')), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1, annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dot", row=2, col=1, annotation_text="Oversold")

    # ADI
    fig.add_trace(go.Scatter(x=df.index, y=df['ADI'], name="ADI", line=dict(color='blue')), row=3, col=1)

    fig.update_layout(title=f"Analisi Tecnica - {ticker}", height=900)
    fig.write_html(f"chart_{ticker}_{datetime.now().strftime('%Y%m%d')}.html")
    print("✅ Grafico salvato come file HTML (aprilo nel browser)")


def main():
    try:
        data = load_data()
        data = calculate_indicators(data)
        data = generate_signals(data)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Salvataggio
        data.to_csv(f"data_with_signals_{timestamp}.csv")
        essential = data[['open','high','low','close','volume','ADI','RSI','SMA_20','EMA_9','MACD','Signal']]
        essential.to_csv(f"data_essential_signals_{timestamp}.csv")

        print(f"\n✅ File salvati con successo!")

        # Grafico
        create_chart(data, TICKER if DATA_SOURCE == "yfinance" else "Asset")

        print("\nAnteprima ultimi 10 segnali:")
        print(data[['close', 'RSI', 'ADI', 'Signal']].tail(10))

    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        print("Verifica il file CSV, il ticker o le dipendenze installate.")


if __name__ == "__main__":
    main()
