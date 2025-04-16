import yfinance as yf
import pandas as pd

def load_data(ticker):
    df = yf.download(ticker, period="1y", interval="1d", group_by="column", auto_adjust=False)

    # If it's a MultiIndex (like with yfinance multi-ticker format), flatten it
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)  # Just take the data level (e.g., "High", "Low", etc.)

    print("Flattened columns:", df.columns)

    # Ensure required columns exist
    if df.empty or 'High' not in df.columns or 'Low' not in df.columns:
        return None

    # Compute indicators
    df["SMA_20"] = df["High"].rolling(window=20).mean()
    df["SMA_50"] = df["High"].rolling(window=50).mean()

    delta = df["High"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)
    return df
