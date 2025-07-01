import os
import time
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

# ---------------- SETTING API KEY (bisa dari .env atau hardcode) ---------------
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

client = Client(API_KEY, API_SECRET, {"timeout": 30})

def fetch_and_append_btcusdt_4h(
    client: Client,
    csv_path: str = "BTCUSDT.csv",
    symbol: str = "BTCUSDT",
    interval: str = Client.KLINE_INTERVAL_4HOUR,
    start_str: str = "1 Jan 2017",
    max_retries: int = 5
):
    # Cek apakah file sudah ada
    if os.path.isfile(csv_path):
        df_old = pd.read_csv(csv_path, parse_dates=["open_time"])
        last_time = df_old["open_time"].max()
        start_time = last_time + pd.Timedelta(hours=4)
        print(f"[INFO] Data lama ditemukan. Mulai fetch dari {start_time}")
        start_str = start_time.strftime("%d %b %Y %H:%M:%S")
    else:
        print(f"[INFO] File {csv_path} belum ada. Mulai fetch dari awal: {start_str}")
        df_old = pd.DataFrame()

    all_data = []
    while True:
        for attempt in range(max_retries):
            try:
                klines = client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=start_str,
                    end_str=None,
                    limit=1000
                )
                break  # sukses, keluar dari loop retry
            except (BinanceAPIException, BinanceRequestException, Exception) as e:
                print(f"[ERROR] {e}, retry {attempt+1}/{max_retries}")
                time.sleep(2 * (attempt+1))
        else:
            print("[FAIL] Tidak bisa fetch dari Binance setelah beberapa percobaan.")
            break

        if not klines:
            print("[DONE] Tidak ada data baru dari Binance.")
            break

        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        
        all_data.append(df)
        last_open = df["open_time"].max()
        now = pd.Timestamp.utcnow()

        print(f"[INFO] Last open fetched: {last_open}, now: {now}")

        if last_open >= now - pd.Timedelta(hours=4):
            print("[COMPLETE] Data sudah up-to-date.")
            break
        # Set start_str untuk iterasi berikutnya
        start_str = (last_open + pd.Timedelta(hours=4)).strftime("%d %b %Y %H:%M:%S")
        time.sleep(1.5)

    if all_data:
        df_new = pd.concat(all_data)
        if not df_old.empty:
            df_all = pd.concat([df_old, df_new]).drop_duplicates(subset="open_time").sort_values("open_time")
        else:
            df_all = df_new.sort_values("open_time")
        df_all.to_csv(csv_path, index=False)
        print(f"[OK] Data BTCUSDT diperbarui sampai {df_all['open_time'].max()}")
    else:
        print("[WARNING] Tidak ada data baru yang ditambahkan.")

if __name__ == "__main__":
    fetch_and_append_btcusdt_4h(client)
