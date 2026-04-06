"""
Strategia di Analisi Tecnica per Opzioni Binarie
- Caricamento dati
- Pulizia e preparazione
- Calcolo sicuro dell'Accumulation/Distribution Index (ADI)
- Evita gli errori di pandas_ta (1-dimensional, sanitize_array, ecc.)
"""

import pandas as pd
import pandas_ta as ta
from pathlib import Path

# ========================= CONFIGURAZIONE =========================
# Cambia qui il percorso del tuo file CSV (o usa yfinance, ecc.)
DATA_PATH = Path("data.csv")          # <-- MODIFICA CON IL TUO FILE

# Nome delle colonne nel tuo CSV (modificale se necessario)
COLUMN_MAPPING = {
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume',
    # Aggiungi altre colonne se servono
}

# ================================================================

def load_and_prepare_data(file_path: Path) -> pd.DataFrame:
    """Carica i dati e li prepara in formato corretto."""
    print(f"Caricamento dati da: {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"File non trovato: {file_path}")
    
    # Carica il CSV (modifica delimiter o parse_dates se serve)
    df = pd.read_csv(file_path)
    
    # Rinomina colonne in minuscolo (standard pandas_ta)
    df = df.rename(columns=COLUMN_MAPPING)
    
    # Converti in numerico (sicuro)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Gestione indice temporale (molto importante per evitare errori!)
    if 'Date' in df.columns or 'date' in df.columns:
        date_col = 'Date' if 'Date' in df.columns else 'date'
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    
    df = df.sort_index()
    
    # Rimuovi righe con valori mancanti nelle colonne essenziali
    required_cols = ['high', 'low', 'close', 'volume']
    df = df.dropna(subset=required_cols)
    
    print(f"Dati caricati con successo: {len(df)} righe")
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcola gli indicatori in modo sicuro (evita errori comuni di pandas_ta)."""
    print("Calcolo indicatori tecnici...")
    
    # Forza le colonne a essere Series 1-dimensionali (risolve l'errore "Data must be 1-dimensional")
    df = df.copy()
    for col in ['high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = df[col].squeeze()   # rimuove dimensioni extra
    
    # Calcolo esplicito e sicuro dell'ADI (Accumulation/Distribution Index)
    try:
        df['ADI'] = ta.ad(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume']
        )
        print("✅ Accumulation/Distribution Index calcolato correttamente")
    except Exception as e:
        print(f"❌ Errore durante il calcolo ADI: {e}")
        raise
    
    # Qui puoi aggiungere altri indicatori che ti servono (esempi):
    # df['SMA_20'] = ta.sma(df['close'], length=20)
    # df['RSI'] = ta.rsi(df['close'], length=14)
    
    return df


def main():
    try:
        # 1. Carica e prepara i dati
        data = load_and_prepare_data(DATA_PATH)
        
        # 2. Calcola gli indicatori in modo robusto
        data = calculate_indicators(data)
        
        # 3. Salva il risultato
        output_file = "data_with_adi.csv"
        data.to_csv(output_file)
        print(f"✅ File salvato con successo: {output_file}")
        
        # Mostra un'anteprima
        print("\nAnteprima degli ultimi dati:")
        print(data[['close', 'volume', 'ADI']].tail(10))
        
    except Exception as e:
        print(f"\n❌ Errore generale: {e}")
        print("Controlla il percorso del file e i nomi delle colonne.")


if __name__ == "__main__":
    main()
