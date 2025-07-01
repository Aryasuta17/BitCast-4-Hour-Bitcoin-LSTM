# ⚡ Bitcast: Bitcoin Price Forecasting & Analytics

Bitcast adalah web app prediksi harga Bitcoin berbasis deep learning yang dikembangkan sebagai project personal. Bitcast memadukan data engineering, machine learning, dan otomatisasi—menyediakan visualisasi tren harga BTC dan prediksi secara real-time melalui antarmuka yang modern dan interaktif dimana kami memiliki pretrain yang dapat digunakan dalam interval data 4 jam tujuan kami adalah untuk mendapatkan bitcoin price prediction dalam interval waktuu 4 jam.

---

## 🚀 Fitur Utama

- **Prediksi Harga Bitcoin Real-Time**
  - Model LSTM untuk forecasting 1–30 hari ke depan.
- **Otomasi Fetch Data & Retrain Model**
  - Data harga otomatis diambil dan model diretrain berkala, memastikan prediksi selalu up-to-date.
- **Dashboard Visual Interaktif**
  - Grafik prediksi vs. harga aktual, tren historis, dan analisis error.
- **REST API**
  - Endpoint untuk integrasi prediksi ke aplikasi lain.
- **Analisis Error**
  - MSE, RMSE, MAE ditampilkan transparan untuk evaluasi akurasi model.
- **Parameter Custom**
  - Range prediksi, jadwal retrain, dll. bisa dikustomisasi.

---

## ⚙️ Stack Teknologi

- **Backend:** Python, Flask, TensorFlow/Keras, Pandas, APScheduler
- **Frontend:** HTML, CSS (Tailwind/Bootstrap), Chart.js/Plotly
- **Data Source:** Binance API / CSV
- **Deployment:** Local server, Docker-ready

---

## 📊 Cara Kerja

1. **Data Pipeline**  
   Mengambil dan menyiapkan data harga BTC/USDT dari Binance.
2. **LSTM Deep Learning Model**  
   Melatih model pada data historis untuk mengenali pola dan prediksi harga ke depan.
3. **Scheduled Automation**  
   Model & data otomatis di-refresh berkala dengan scheduler (setiap beberapa jam).
4. **Web Interface & API**  
   User dapat melihat dashboard atau request prediksi via API endpoint.

---

## 🧠 Motivasi

Bitcast lahir dari keingintahuan akan prediktabilitas harga Bitcoin, kecintaan pada data science & ML, serta dorongan untuk membuat sistem forecasting yang **transparan dan open-source**.  
Project ini sekaligus jadi playground untuk eksperimen ML ops, time series, dan end-to-end deployment.

---

## 📷 Screenshot

####  Tampilan Awal Data Real Time
![Bitcast Dashboard Example](assets/bitcast_dashboard2.png)  

####  Tampilan Awal Prediksi
![Bitcast Dashboard Example](assets/bitcast_dashboard2.png)  

---

## 🚦 Cara Menjalankan

```bash
git clone https://github.com/Aryasuta17/bitcast.git
cd bitcast
pip install -r requirements.txt
python app.py
