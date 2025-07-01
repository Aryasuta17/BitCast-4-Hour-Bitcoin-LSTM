from flask import Flask, render_template, jsonify, request
import requests
import time
import logging
from model_utils import get_btc_forecast, retrain_model, fetch_new_4h_data
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import pytz
import os
import pandas as pd
import threading

app = Flask(__name__)

JAKARTA_TZ = pytz.timezone("Asia/Jakarta")

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/api/predict/<symbol>")
def api_predict(symbol):
    symbol = symbol.upper()
    if symbol != "BTC":
        return jsonify({"error": "only BTC supported"}), 400
    try:
        dates, preds, last_close = get_btc_forecast()
        print("API PREDICT DEBUG", dates[:3], "...", dates[-3:])  # Cek 3 awal & akhir
        print("API PREDICT DEBUG pred:", preds[:3], "...", preds[-3:])
        print("API PREDICT DEBUG last_close:", last_close)
        return jsonify(
            {"symbol": "BTC", "current": last_close, "dates": dates, "prices": preds}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/live-price/<symbol>")
def api_live_price(symbol):
    """Get current live price from Binance"""
    symbol = symbol.upper()
    try:
        resp = requests.get(f'https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT', timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return jsonify({
            "symbol": symbol,
            "price": float(data['price']),
            "timestamp": int(time.time() * 1000)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/exchange')
def exchange():
    return render_template('exchange.html')

@app.route('/prediksi')
def prediksi():
    return render_template('prediksi.html')

@app.route('/coin/<symbol>')
def coin_chart(symbol):
    return render_template('coin.html', symbol=symbol.upper())
    
@app.route('/model')
def model():
    return render_template('model.html')

@app.route("/hasil")
def hasil():
    symbol = request.args.get("symbol", "BTC")
    return render_template("hasil.html", symbol=symbol)

@app.route('/tentang')
def tentang():
    return render_template('tentang.html')

def fetch_binance_data():
    try:
        resp = requests.get('https://api.binance.com/api/v3/ticker/24hr', timeout=10)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        logging.error(f"Binance API error: {e}")
        return []

    coins = []
    for item in raw:
        sym = item.get('symbol', '')
        if not sym.endswith('USDT'):
            continue
        try:
            base = sym[:-4].upper()
            coins.append({
                'symbol': base,
                'price': float(item['lastPrice']),
                'change_pct': float(item['priceChangePercent']),
                'volume': float(item['quoteVolume']),
            })
        except:
            continue

    coins.sort(key=lambda x: x['volume'], reverse=True)

    gecko_caps = {}
    for page in [1, 2]:
        try:
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': 250,
                'page': page,
                'sparkline': 'false'
            }
            r = requests.get('https://api.coingecko.com/api/v3/coins/markets', params=params, timeout=10)
            r.raise_for_status()
            for c in r.json():
                # symbol di CoinGecko adalah ticker lowercase, e.g. "btc"
                gecko_caps[c['symbol'].upper()] = c.get('market_cap', None)
            time.sleep(1)  # hindari rate limit
        except Exception as e:
            logging.warning(f"CoinGecko page {page} failed: {e}")

    for c in coins:
        c['market_cap'] = gecko_caps.get(c['symbol'], None)

    for i, c in enumerate(coins, start=1):
        c['rank'] = i

    return coins

@app.route('/api/klines')
def api_klines():
    symbol = request.args.get('symbol', 'BTC').upper()
    interval = request.args.get('interval', '15m')
    
    # Validasi interval
    valid_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    if interval not in valid_intervals:
        interval = '15m'
    
    # Tentukan limit berdasarkan interval
    limit_map = {
        '1m': 100, '3m': 100, '5m': 100, '15m': 100, '30m': 100,
        '1h': 168, '2h': 168, '4h': 168, '6h': 168, '8h': 168, '12h': 168,
        '1d': 30, '3d': 30, '1w': 52, '1M': 12
    }
    limit = limit_map.get(interval, 100)
    
    try:
        # Panggil Binance Klines API
        url = 'https://api.binance.com/api/v3/klines'
        params = {
            'symbol': f'{symbol}USDT',
            'interval': interval,
            'limit': limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Format data untuk ApexCharts
        formatted_data = []
        for item in data:
            # Binance kline format: [timestamp, open, high, low, close, volume, ...]
            formatted_data.append({
                'x': int(item[0]),  # timestamp
                'open': float(item[1]),
                'high': float(item[2]),
                'low': float(item[3]),
                'close': float(item[4]),
                'volume': float(item[5])
            })
        
        return jsonify(formatted_data)
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Binance Klines API error: {e}")
        return jsonify([]), 500
    except Exception as e:
        logging.error(f"Error processing klines data: {e}")
        return jsonify([]), 500

@app.route("/api/retrain")
def api_retrain():
    result = retrain_model()
    return jsonify({"status": result})

@app.route('/api/binance')
def api_binance():
    return jsonify(fetch_binance_data())

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store"
    return response

def is_new_batch_ready(batch_hour, data_path="BTCUSDT.csv"):
    if not os.path.exists(data_path):
        return False, None
    df = pd.read_csv(data_path, parse_dates=["open_time"])

    df["open_time"] = pd.to_datetime(df["open_time"]).dt.tz_convert(JAKARTA_TZ)

    target_rows = df[df["open_time"].dt.hour == batch_hour]
    if not target_rows.empty:

        latest_row = target_rows.iloc[-1]
        return True, latest_row
    else:
        return False, None

def retrain_model_with_row(row):

    print(f"[JKT] Model retrained dengan data: {row['open_time']} open={row['open']} close={row['close']}")

def retrain_job():
    print("[JKT] Fetching new data sebelum retrain...")
    updated = fetch_new_4h_data()
    if updated:
        retrain_model()
    else:
        print("[JKT] Tidak ada data baru, skip retrain.")

scheduler = BackgroundScheduler()
scheduler.add_job(
    retrain_job,
    'cron',
    minute=5,
    hour='3,7,11,15,19,23',
    timezone=JAKARTA_TZ
)

scheduler.start()
print("[INFO] Scheduler started for retrain jam 3,7,11,15,19,23 di Asia/Jakarta +5min.")


@app.route("/api/manual-retrain")
def api_manual_retrain():
    updated = fetch_new_4h_data()
    retrain_status = retrain_model()
    return jsonify({
        "fetch_status": "updated" if updated else "no new data",
        "retrain_status": retrain_status
    })

def fetch_and_retrain():
    try:
        print("[JKT] First fetch + retrain on startup (background thread)")
        updated = fetch_new_4h_data()
        if updated:
            retrain_model()
        else:
            print("Tidak ada data baru, retrain dilewati.")
    except Exception as e:
        print(f"Error fetch+retrain startup: {e}")

if __name__ == "__main__":
    # Start Flask server
    # Jalankan fetch + retrain di thread terpisah, web tetap up
    thread = threading.Thread(target=fetch_and_retrain)
    thread.daemon = True
    thread.start()
    app.run(debug=True)

