# preprocess.py
import numpy as np
import pandas as pd

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Mengisi NA kecil (<0.05%) dengan forward-fill
    - Membuat fitur lag dan moving averages untuk kolom 'close'
    - Drop baris yang masih NaN setelah itu
    """
    # isi NA kecil
    for col in df.columns:
        if df[col].isnull().mean() < 0.0005:
            df[col].fillna(method='ffill', inplace=True)
    # fitur lag
    lag_periods = [1, 6, 12, 24, 48]
    for p in lag_periods:
        df[f"lag{p}"] = df['close'].shift(p)
    # moving averages
    ma_windows = [6, 12, 24, 48]
    for w in ma_windows:
        df[f"ma{w}"] = df['close'].rolling(w).mean()
    # buang baris NaN
    df = df.dropna()
    return df

def make_sequences(data: np.ndarray, seq_len: int, horizon: int):
    X, y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len : i + seq_len + horizon, -1])
    return np.array(X), np.array(y)
