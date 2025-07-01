import pickle, math, os
from datetime import datetime, timedelta
import pytz
from binance.client import Client
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
import joblib

# ------------------------------------------------------------------
MODEL_PATH   = "lstm_btc_best.h5"     # final fine-tuned model
SCALER_PATH  = "scaler_btc_best.pkl"                # contains {'scaler','feat','scaler_close'}
RAW_CSV_PATH = "BTCUSDT.csv"               # 4-hour candles (must stay up-to-date)

# static hyper-params (the model was trained with these)
SEQ_LEN   = 258
HORIZON   = 12           # 14 days × (24/4) = 84 steps
STEP_HRS  = 4
# ------------------------------------------------------------------
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
BINANCE_SYMBOL = "BTCUSDT"
CSV_FILE = "BTCUSDT.csv"

FEATURES = [
    'close', 'volume', 'SMA_10', 'EMA_10', 'MACD', 'RSI_14', 'ATR_14', 'OBV', 'Pct_Change'
]

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["close"]; high  = df["high"]
    low    = df["low"];   volume = df["volume"]
    df["SMA_10"] = close.rolling(10).mean()
    df["EMA_10"] = close.ewm(span=10, adjust=False).mean()
    ema12       = close.ewm(span=12, adjust=False).mean()
    ema26       = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    delta      = close.diff()
    gain       = delta.where(delta > 0, 0.0)
    loss       = (-delta).where(delta < 0, 0.0)
    avg_gain   = gain.rolling(14).mean()
    avg_loss   = loss.rolling(14).mean()
    rs         = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()
    obv = [0]
    for i in range(1, len(df)):
        if close.iat[i] > close.iat[i - 1]:
            obv.append(obv[-1] + volume.iat[i])
        elif close.iat[i] < close.iat[i - 1]:
            obv.append(obv[-1] - volume.iat[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv
    df["Pct_Change"] = (close - df["open"]) / df["open"] * 100
    return df

def fetch_new_4h_data():
    import pandas as pd
    from binance.client import Client
    import pytz
    import os
    import time

    # --- Konfigurasi ---
    API_KEY = os.getenv("BINANCE_API_KEY")
    API_SECRET = os.getenv("BINANCE_API_SECRET")
    BINANCE_SYMBOL = "BTCUSDT"
    CSV_FILE = "BTCUSDT.csv"
    JAKARTA_TZ = pytz.timezone("Asia/Jakarta")

    # --- Inisialisasi Client Binance ---
    try:
        client = Client(API_KEY, API_SECRET)
        # test koneksi, jika error langsung raise
        client.ping()
    except Exception as e:
        print(f"[ERROR] Gagal inisialisasi Binance Client: {e}")
        return False

    print("=== MULAI FETCH DEBUG ===")
    print("Baca file:", CSV_FILE)
    if os.path.isfile(CSV_FILE):
        df = pd.read_csv(CSV_FILE, parse_dates=["open_time"], index_col="open_time")
        last_time = df.index[-1]
        # Convert index (open_time) ke Asia/Jakarta jika belum
        if getattr(last_time, "tzinfo", None) is None or last_time.tzinfo.utcoffset(last_time) is None:
            last_time = pd.Timestamp(last_time).tz_localize(JAKARTA_TZ)
        else:
            last_time = last_time.tz_convert(JAKARTA_TZ)
        print("last_time Jakarta:", last_time)
    else:
        df = None
        last_time = pd.Timestamp("2017-01-01 00:00:00", tz=JAKARTA_TZ)
        print("No CSV, mulai dari 2017")

    # Cari jam batch berikutnya (3,7,11,15,19,23)
    possible_batches = [3, 7, 11, 15, 19, 23]
    last_hour = last_time.hour
    next_batch_hour = min([h for h in possible_batches if h > last_hour], default=possible_batches[0])
    # Jika sudah lewat batch terakhir hari ini, lompat ke besok
    if next_batch_hour <= last_hour:
        next_time = (last_time + pd.Timedelta(days=1)).replace(hour=possible_batches[0], minute=0, second=0, microsecond=0)
    else:
        next_time = last_time.replace(hour=next_batch_hour, minute=0, second=0, microsecond=0)
    print("next_time Jakarta:", next_time)

    # Batas pengambilan data: sekarang (Asia/Jakarta)
    now_jkt = pd.Timestamp.now(tz=JAKARTA_TZ)
    if next_time > now_jkt:
        print("Sudah paling update (no new batch candle).")
        return False

    # Convert next_time dan now_jkt ke UTC string untuk Binance
    start_utc = next_time.tz_convert(pytz.UTC)
    end_utc   = now_jkt.tz_convert(pytz.UTC)
    start_str = start_utc.strftime("%d %b %Y %H:%M:%S")
    end_str   = end_utc.strftime("%d %b %Y %H:%M:%S")
    print(f"Fetching from {start_str} UTC to {end_str} UTC")

    # --- Fetch Data dari Binance ---
    try:
        # Batas timeout request internal Binance: (gunakan waktu yang agak lama agar stabil)
        klines = client.get_historical_klines(
            BINANCE_SYMBOL,
            Client.KLINE_INTERVAL_4HOUR,
            start_str,
            end_str
        )
        print(f"klines dapat: {len(klines)} baris")
    except Exception as e:
        print(f"[ERROR] Fetch Binance gagal: {e}")
        return False

    if not klines:
        print("Tidak ada data baru dari Binance.")
        return False

    # Parse ke df
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]
    df_new = pd.DataFrame(klines, columns=cols)
    df_new["open_time"] = pd.to_datetime(df_new["open_time"], unit="ms", utc=True).dt.tz_convert(JAKARTA_TZ)
    df_new = df_new.set_index("open_time").astype(float)
    
    # Asumsikan fungsi compute_technical_indicators(df) dan FEATURES sudah ada/imported
    df_new = compute_technical_indicators(df_new)
    df_new = df_new[FEATURES]

    # Gabungkan data baru ke file lama (hindari duplikat)
    if df is not None:
        df_final = pd.concat([df, df_new])
        df_final = df_final[~df_final.index.duplicated()]
    else:
        df_final = df_new

    df_final.to_csv(CSV_FILE)
    print(f"Data baru ditambahkan sampai {df_final.index[-1]}")
    return True

def retrain_model(seq_len=258, horizon=12, epochs=5):
    import pandas as pd
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    from sklearn.preprocessing import MinMaxScaler
    import pickle

    print("==== MULAI RETRAIN MODEL ====")
    # 1. Load data
    df = pd.read_csv("BTCUSDT.csv", parse_dates=["open_time"], index_col="open_time")
    df = df.dropna()
    feat = ['close', 'volume', 'SMA_10', 'EMA_10', 'MACD', 'RSI_14', 'ATR_14', 'OBV', 'Pct_Change']
    target = "close"

    # 2. Fit scaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[feat])
    scaler_close = MinMaxScaler()
    y_scaled = scaler_close.fit_transform(df[[target]])

    # 3. Build sequence dataset
    Xs, Ys = [], []
    for i in range(len(X_scaled) - seq_len - horizon + 1):
        Xs.append(X_scaled[i:i+seq_len])
        Ys.append(y_scaled[i+seq_len:i+seq_len+horizon, 0])
    Xs, Ys = np.array(Xs), np.array(Ys)

    print(f"Dataset siap: Xs {Xs.shape}, Ys {Ys.shape}")
    if len(Xs) == 0:
        print("ERROR: Sequence dataset kosong, tidak cukup data!")
        return "ERROR: Tidak cukup data untuk retrain"

    # 4. Build model
    model = Sequential([
        Input(shape=(seq_len, len(feat))),
        LSTM(64, return_sequences=False),
        Dense(horizon)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xs, Ys, epochs=epochs, batch_size=32, verbose=1)
    model.save("lstm_btc_best.h5")
    print("Model berhasil disimpan ke lstm_btc_best.h5")

    obj = {"scaler": scaler, "feat": feat, "scaler_close": scaler_close}
    with open("scaler_btc_best.pkl", "wb") as f:
        pickle.dump(obj, f)
    print("Scaler berhasil disimpan ke scaler_btc_best.pkl")

    print("==== RETRAIN MODEL SELESAI ====")
    return "Retrain selesai"

def _load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        raise FileNotFoundError("Model or scaler pickle not found – train first!")

    print("Loading model from:", MODEL_PATH)
    model = load_model(MODEL_PATH, compile=False)

    try:
        print("Loading scaler from:", SCALER_PATH)
        with open(SCALER_PATH, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load scaler: {e}")

    if not isinstance(obj, dict):
        raise TypeError("Scaler file does not contain a valid dictionary.")

    required_keys = {"scaler", "feat", "scaler_close"}
    if not required_keys.issubset(obj):
        raise KeyError(f"Scaler file missing required keys: {required_keys - obj.keys()}")

    scaler, feat, scaler_close = obj["scaler"], obj["feat"], obj["scaler_close"]
    return model, scaler, feat, scaler_close

import pandas as pd
from datetime import timedelta
import numpy as np

# ... (import dan function lain sesuai kode kamu di atas)

def get_btc_forecast():
    # Load model dan scaler dulu!
    model, scaler, feat, scaler_close = _load_artifacts()

    # Load data
    df_raw = pd.read_csv(RAW_CSV_PATH, parse_dates=["open_time"], index_col="open_time")

    anchor_time = df_raw.index[-1]
    print("[DEBUG] anchor_time (fix):", anchor_time, type(anchor_time))

    df = df_raw.copy()
    df = df.ffill().bfill()  # Isi NA dengan data terdekat

    lags = [1, 2, 3, 6, 12]
    mas  = [3, 6, 12, 18]
    for p in lags:
        df[f"lag{p}"] = df["close"].shift(p)
    for w in mas:
        df[f"ma{w}"] = df["close"].rolling(w).mean()
    df = df.ffill().bfill()

    df_window = df[-SEQ_LEN:]
    values = scaler.transform(df_window[feat])
    last_seq = values.reshape(1, SEQ_LEN, len(feat))

    # Prediksi
    future_scaled = model.predict(last_seq, verbose=0)[0]
    future_prices = scaler_close.inverse_transform(
        future_scaled.reshape(-1, 1)
    ).flatten().tolist()

    # Future dates dari anchor_time
    step = timedelta(hours=STEP_HRS)
    future_dates = [
        (anchor_time + step * (i + 1)).isoformat() for i in range(HORIZON)
    ]
    last_close_price = float(df_raw["close"].iloc[-1])

    print("[DEBUG] last_close_price:", last_close_price)
    print("[DEBUG] future_dates:", future_dates)
    print("[DEBUG] future_prices:", future_prices[:3], "...", future_prices[-3:])

    return future_dates, future_prices, last_close_price





