import streamlit as st
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import datetime

st.set_page_config(page_title="Weather Forecast Regression", layout="wide")

# Sidebar - Input user
st.sidebar.title("Prediksi Suhu Masa Depan")
future_date = st.sidebar.date_input("Pilih Tanggal", datetime.date(2025, 5, 7))
hour = st.sidebar.slider("Jam (0–23)", 0, 23, 12)
precipitation = st.sidebar.number_input("Presipitasi (mm)", min_value=0.0, step=0.1)

# Title
st.title("📊 Multivariate Forecasting Regression - Weather Data")

# --- Fetch historical weather data ---
@st.cache_data
def load_data():
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 0.133333,
        "longitude": 117.500000,
        "start_date": "2019-01-01",
        "end_date": "2024-12-31",
        "hourly": ["temperature_2m", "precipitation"],
        "timezone": "Asia/Makassar"
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame({
        "datetime": pd.to_datetime(data["hourly"]["time"]),
        "temperature": data["hourly"]["temperature_2m"],
        "precipitation": data["hourly"]["precipitation"]
    })
    return df

df = load_data()

# Feature engineering
df.set_index("datetime", inplace=True)
df["hour"] = df.index.hour
df["day_of_year"] = df.index.dayofyear
df["month"] = df.index.month

# Show sample data
with st.expander("📈 Tampilkan Data Historis"):
    st.dataframe(df.head())

# Split data
X = df[["hour", "day_of_year", "month", "precipitation"]]
y = df["temperature"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression(),
}

results = {}
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    mae = mean_absolute_error(y_test, y_pred)
    results[name] = mae

# Display model performance
st.subheader("📊 Evaluasi Model")
for name, mae in results.items():
    st.write(f"**{name}** - MAE: {mae:.2f}°C")

# Plot actual vs predicted
selected_model = min(results, key=results.get)
best_model = models[selected_model]
st.success(f"✅ Model Terbaik: {selected_model}")

fig, ax = plt.subplots()
ax.plot(y_test.values[:100], label='Actual')
ax.plot(predictions[selected_model][:100], label='Predicted')
ax.set_title('Prediksi vs Aktual')
ax.set_ylabel('Suhu (°C)')
ax.legend()
st.pyplot(fig)

# --- Predict future temperature ---
st.subheader("📅 Prediksi Suhu untuk Tanggal Dipilih")
future_input = pd.DataFrame({
    "hour": [hour],
    "day_of_year": [future_date.timetuple().tm_yday],
    "month": [future_date.month],
    "precipitation": [precipitation]
})
pred_temp = best_model.predict(future_input)[0]
st.metric(label=f"Prediksi Suhu pada {future_date} pukul {hour}:00", value=f"{pred_temp:.2f}°C")

# --- Fetch actual forecast for comparison ---
@st.cache_data
def get_actual_temp(date):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 0.133333,
        "longitude": 117.500000,
        "hourly": ["temperature_2m"],
        "timezone": "Asia/Makassar",
        "start_date": date.strftime("%Y-%m-%d"),
        "end_date": date.strftime("%Y-%m-%d")
    }
    response = requests.get(url, params=params)
    return response.json()

actual_data = get_actual_temp(future_date)
if "hourly" in actual_data and "temperature_2m" in actual_data["hourly"]:
    actual_temp = actual_data["hourly"]["temperature_2m"][hour]
    error = abs(actual_temp - pred_temp)
    st.info(f"🌡️ Suhu Aktual: {actual_temp:.2f}°C")
    st.write(f"📉 Perbedaan dengan prediksi: {error:.2f}°C")
else:
    st.warning("⚠️ Data suhu aktual belum tersedia untuk tanggal tersebut.")
