import streamlit as st
import pandas as pd
import requests
import joblib
from datetime import datetime
import io

# --- Load model dari GitHub ---
MODEL_URL = "https://raw.githubusercontent.com/rootAmr/predict_cuaca_bontang/main/best_weather_model.pkl"

try:
    response = requests.get(MODEL_URL)
    response.raise_for_status()
    model_bytes = io.BytesIO(response.content)
    best_model = joblib.load(model_bytes)
    st.success("✅ Model berhasil dimuat dari GitHub!")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# --- Sidebar input untuk pengguna ---
st.sidebar.title("Prediksi Suhu Masa Depan")
future_date = st.sidebar.date_input("Pilih Tanggal", datetime(2025, 5, 7))
hour = st.sidebar.slider("Pilih Jam (0-23)", 0, 23, 12)
precipitation = st.sidebar.number_input("Presipitasi (mm)", min_value=0.0, step=0.1)

# --- Tampilkan judul aplikasi ---
st.title("📊 Multivariate Forecasting Regression - Weather Data")

# --- Buat fitur untuk prediksi masa depan ---
future_features = pd.DataFrame({
    "hour": [hour],
    "day_of_year": [future_date.timetuple().tm_yday],
    "month": [future_date.month],
    "precipitation": [precipitation]
})

# --- Prediksi suhu ---
pred_temp = best_model.predict(future_features)[0]
st.metric(label=f"Prediksi Suhu untuk {future_date} pukul {hour}:00", value=f"{pred_temp:.2f}°C")

# --- Suhu aktual dari API ---
actual_url = "https://api.open-meteo.com/v1/forecast"
actual_params = {
    "latitude": 0.133333,
    "longitude": 117.500000,
    "hourly": ["temperature_2m"],
    "timezone": "Asia/Makassar",
    "start_date": future_date.strftime("%Y-%m-%d"),
    "end_date": future_date.strftime("%Y-%m-%d")
}
actual_response = requests.get(actual_url, params=actual_params)
actual_data = actual_response.json()

if "hourly" in actual_data and "temperature_2m" in actual_data["hourly"]:
    actual_temp = actual_data["hourly"]["temperature_2m"][hour]
    error = abs(actual_temp - pred_temp)
    st.info(f"🌡️ Suhu Aktual: {actual_temp:.2f}°C")
    st.write(f"📉 Perbedaan dengan prediksi: {error:.2f}°C")
else:
    st.warning("⚠️ Data suhu aktual belum tersedia untuk tanggal tersebut.")
