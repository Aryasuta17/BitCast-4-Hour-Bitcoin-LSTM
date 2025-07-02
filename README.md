# ⚡ Bitcast: Bitcoin Price Forecasting & Analytics

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Framework](https://img.shields.io/badge/Framework-Flask-red)
![Model](https://img.shields.io/badge/Model-LSTM-purple)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Forecasting](https://img.shields.io/badge/Multivariate-Forecasting-red)

---

## 🧩 Overview

**Bitcast** is an end-to-end deep learning-based Bitcoin price forecasting web application, designed with a clean and interactive interface for both individual users and system integrators. Built entirely as a personal side project, Bitcast embodies the fusion of **data engineering**, **machine learning**, and **real-time automation**—delivering BTC/USDT price predictions and analytics in a scalable and modular fashion.

The system utilizes **LSTM (Long Short-Term Memory)** architecture to predict Bitcoin prices from 1 up to 30 days ahead, trained on multivariate time series data sourced directly from Binance. It supports **automated data refresh**, **scheduled retraining**, and **REST API exposure**, making it suitable as a foundation for real-world financial applications or as a learning tool for time series enthusiasts.

---

## 🚀 Key Features

### 📈 Real-Time Bitcoin Forecasting
- Predict BTC/USDT price using an LSTM deep learning model.
- Supports forecasting for 1–30 days.
- Pretrained model optimized for 4-hour interval data.

### 🔁 Automated Data Pipeline & Retraining
- Automatically fetches the latest Bitcoin price data via Binance API.
- Triggers model retraining periodically using `APScheduler`.
- Ensures the model remains adaptive and updated.

### 📊 Interactive Dashboard Interface
- Intuitive visualization of:
  - Predicted vs actual price curves.
  - Historical BTC price trends.
  - Evaluation metrics over time.

### 🌐 Public REST API
- Expose prediction functionality via `/api/predict/BTC`.
- Enables easy integration with trading bots, dashboards, or external systems.

### 📉 Transparent Error Metrics
- Model evaluation includes:
  - **Mean Squared Error (MSE)**
  - **Root Mean Squared Error (RMSE)**
  - **Mean Absolute Error (MAE)**
- Helps quantify prediction performance over time.

### ⚙️ Customizable Parameters
- Configure:
  - Prediction window (1–30 days)
  - Retraining frequency
  - Input data granularity
  - Model hyperparameters

---

## 📈 Model Performance

The current model was trained on historical 4-hour interval BTC/USDT data.  
Performance on test data (unseen during training):

- **RMSE:** `186.92 USDT`
- **MAE:** `132.34 USDT`

While crypto price prediction is notoriously volatile and nonlinear, Bitcast aims to provide a **probabilistic view of trend direction** rather than exact price targets.

---

## ⚙️ Tech Stack

| Layer         | Tools Used                                              |
|---------------|---------------------------------------------------------|
| **Backend**   | Python, Flask, TensorFlow/Keras, Pandas, NumPy         |
| **Frontend**  | HTML5, CSS3, Tailwind/Bootstrap, Chart.js or Plotly    |
| **Automation**| APScheduler for background scheduling tasks            |
| **Data Source**| Binance API (4-hour intervals) + historical CSV backup|
| **Deployment**| Localhost ready, Docker-ready architecture             |

---

## 🔧 System Architecture

```text
+--------------------+                +-----------------------------+                +-------------+
|      Mulai         | ─────────────> |     Model Development       | ─────────────> | Web Dev     |
+--------------------+                +-----------------------------+                +-------------+

Model Development:
  ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
  │  Data Collection                                                                             │
  │    └── Binance & CoinGecko                                                                   │
  │         ↓                                                                                    │
  │  Preprocessing                                                                               │
  │    └── Feature Extraction: SMA, EMA, MACD, RSI, ATR, OBV, Percent Change                     │
  │         ↓                                                                                    │
  │  Feature Selection & Scaling                                                                 │
  │    └── MinMaxScaler                                                                          │
  │         ↓                                                                                    │
  │  Time Series Cross Validation                                                                │
  │    └── 10-Fold CV                                                                            │
  │         ↓                                                                                    │
  │  Setup CPU/GPU Strategy                                                                      │
  │         ↓                                                                                    │
  │  Grid Search                                                                                 │
  │    └── Loop: units, dropout, lr                                                              │
  │    └── Callbacks: EarlyStopping, ReduceLROnPlateau                                           │
  │         ↓                                                                                    │
  │  Check Convergence? ────No────> Return to Grid Search                                        │
  │         │                                                                                    │
  │        Yes                                                                                   │
  │         ↓                                                                                    │
  │  Final Evaluation                                                                            │
  │    └── Metrics: MSE, RMSE, MAE                                                               │
  │    └── Prediction Plot                                                                       │
  └──────────────────────────────────────────────────────────────────────────────────────────────┘

Web Development:
  ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
  │  Backend Design                                                                              │
  │    └── Routes: /live, /predict                                                               │
  │         ↓                                                                                    │
  │  API Integration                                                                             │
  │    └── Connect Fetcher + Model + API                                                         │
  │         ↓                                                                                    │
  │  Frontend Templates                                                                          │
  │    └── index.html, prediksi.html, exchange.html, tentang.html                                │
  │         ↓                                                                                    │
  │  Static Assets                                                                               │
  │    └── CSS, JS, Images                                                                       │
  │         ↓                                                                                    │
  │  Testing & Debug                                                                             │
  │         ↓                                                                                    │
  │  Deployment                                                                                  │
  │         ↓                                                                                    │
  │  Selesai                                                                                     │
  └──────────────────────────────────────────────────────────────────────────────────────────────┘
```
## 🧠 Why Bitcast?
---
### “True intelligence is being able to build systems that learn over time—without constant human tuning.”

This project began from curiosity: Can Bitcoin prices be meaningfully predicted using deep learning models?
--
Bitcast was built as:
---
- A challenge to build an automated machine learning system end-to-end.
- A practical implementation of time series forecasting in a real-world domain.
- A playground to test LSTM performance under volatile financial conditions.
- A foundation for potentially more advanced hybrid models (e.g., LSTM+ARIMA, transformers, or attention-based architectures in future).
---
## 📍 Real-Time Dashboard Overview
![Bitcast Dashboard Example](assets/bitcast_dashboard2.png)  
---

## 🔮 Prediction Graphs
![Bitcast Dashboard Example](assets/bitcast_dashboard2.png)  
---
## 🧪 Getting Started
- 🔧 Prerequisites
- Python 3.10+
- Virtual environment (venv) recommended
---
## 🚀 Installation
```bash
git clone https://github.com/Aryasuta17/bitcast.git
cd bitcast
pip install -r requirements.txt
python app.py
```
## Once running, open your browser and navigate to:
```bash
http://localhost:5000
```
## 🔄 API Usage
---
### 🔹 Get Forecast (BTC)
```bash
GET /api/predict/BTC
```
--
### Returns:
```text
{
  "dates": ["2025-07-02", "2025-07-03", "..."],
  "predictions": [61823.4, 62041.1, "..."],
  "last_close": 61600.9
}
```

## 🔬 Future Improvements
---
- Add attention-based models for better temporal dependency capture
- Multicoin forecasting (ETH, BNB, etc.)
- Incorporate macroeconomic indicators (CPI, interest rate, news sentiment)
- Deploy on cloud (Render, Railway, or AWS)
- Telegram/WhatsApp bot integration for alerts

## 🔗 Contact
---
Made with passion by Aryasuta & Friends

“Bitcast is more than just a predictor it's a lab to shape the frontier of data driven finance.”





