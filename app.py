import streamlit as st
import numpy as np
import pandas as pd
import xarray as xr
import json
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =====================================================
# 1️⃣ CONFIGURATION
# =====================================================
DATA_PATH = "/content/drive/MyDrive/Python Project/IMDAA_1990_2020.nc"
CITY_FILE = "/content/drive/MyDrive/Python Project/user_cities.json"

st.set_page_config(page_title="ClimateGuardX - LSTM Forecast", layout="wide")
st.title("🌦️ ClimateGuardX – LSTM Forecast Dashboard")
st.markdown("Localized LSTM-based climate forecasting with persistent city storage.")

# =====================================================
# 2️⃣ LOAD / SAVE CITY JSON
# =====================================================
def load_city_data():
    """Load coordinates from user_cities.json; create empty if missing."""
    if os.path.exists(CITY_FILE):
        with open(CITY_FILE, "r") as f:
            try:
                user_cities = json.load(f)
                # Ensure correct tuple format
                coords = {c.title(): tuple(v) for c, v in user_cities.items()}
            except Exception:
                st.warning("⚠️ Corrupted JSON file detected. Reinitializing.")
                coords = {}
    else:
        coords = {}
        with open(CITY_FILE, "w") as f:
            json.dump(coords, f)
    return coords

def save_city_data(coords):
    """Save updated coordinates to user_cities.json."""
    normalized = {c.title(): [float(lat), float(lon)] for c, (lat, lon) in coords.items()}
    with open(CITY_FILE, "w") as f:
        json.dump(normalized, f, indent=2)

# Load cities
coords = load_city_data()

# =====================================================
# 3️⃣ LOAD CLIMATE DATASET
# =====================================================
@st.cache_data
def load_dataset():
    """Safely load NetCDF dataset with fallback for engine errors."""
    try:
        # Try normal way first (fastest)
        ds = xr.open_dataset(DATA_PATH)
    except OSError:
        st.warning("⚠️ netCDF4 failed – retrying with h5netcdf engine...")
        try:
            ds = xr.open_dataset(DATA_PATH, engine="h5netcdf")
        except Exception as e:
            st.error(f"❌ Failed to open dataset: {e}")
            st.stop()

    var_descriptions = {
        "HGT_prl": "Geopotential Height (Pressure Level)",
        "TMP_prl": "Temperature (Pressure Level)",
        "TMP_2m": "Surface Air Temperature (2m)",
        "APCP_sfc": "Total Precipitation",
    }

    st.success("✅ Dataset loaded successfully!")
    return ds, var_descriptions

ds, var_descriptions = load_dataset()

# =====================================================
# 4️⃣ HELPER FUNCTIONS
# =====================================================
def extract_timeseries(ds, var, lat, lon):
    lat_idx = np.abs(ds['lat'].values - lat).argmin()
    lon_idx = np.abs(ds['lon'].values - lon).argmin()
    ts = ds[var].isel(latitude=lat_idx, longitude=lon_idx).to_pandas()
    if "units" in ds[var].attrs and "K" in ds[var].attrs["units"]:
        ts = ts - 273.15
    return ts.dropna()

def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def lstm_forecast(series, seq_length=30, steps=30):
    data = series.values.reshape(-1, 1)
    scaler = MinMaxScaler((0, 1))
    scaled = scaler.fit_transform(data)
    X, y = create_sequences(scaled, seq_length)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)

    y_pred = scaler.inverse_transform(model.predict(X_test))
    y_test_inv = scaler.inverse_transform(y_test)
    mae = mean_absolute_error(y_test_inv, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred))

    last_seq = scaled[-seq_length:]
    forecast_scaled = []
    for _ in range(steps):
        pred = model.predict(last_seq.reshape(1, seq_length, 1))
        forecast_scaled.append(pred[0, 0])
        last_seq = np.append(last_seq[1:], pred[0, 0])
    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1))
    forecast_index = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=steps)
    return pd.Series(forecast.flatten(), index=forecast_index), mae, rmse

# =====================================================
# 5️⃣ STREAMLIT UI
# =====================================================
st.sidebar.header("📍 City Input")

# Display known cities
if coords:
    st.sidebar.write("**Saved Cities:**")
    st.sidebar.write(", ".join(coords.keys()))
else:
    st.sidebar.info("No cities saved yet. Add one below.")

city_input = st.sidebar.text_input("Enter City Name (e.g., Delhi, Kanpur, etc.):").strip()
var = st.sidebar.selectbox("Select Climate Variable:", list(var_descriptions.keys()), index=2)
steps = st.sidebar.slider("Forecast Horizon (Days)", 10, 180, 30)

# Handle input
if not city_input:
    st.sidebar.info("Enter a city name to begin.")
    st.stop()

city_key = city_input.strip().lower()
matched_city = [c for c in coords.keys() if c.lower() == city_key]

if matched_city:
    city = matched_city[0]
    lat, lon = coords[city]
    st.sidebar.success(f"✅ Found: {city} ({lat:.2f}, {lon:.2f})")
    has_coords = True
else:
    st.sidebar.warning(f"⚠️ '{city_input}' not found in file. Please enter coordinates:")
    lat = st.sidebar.number_input("Enter Latitude:", format="%.4f")
    lon = st.sidebar.number_input("Enter Longitude:", format="%.4f")
    has_coords = lat != 0 and lon != 0

# =====================================================
# 6️⃣ GENERATE FORECAST BUTTON
# =====================================================
if st.sidebar.button("🚀 Generate Forecast"):
    if not has_coords:
        st.sidebar.error("Please enter valid latitude and longitude values.")
        st.stop()

    # Add new city automatically to JSON if not found
    if all(c.lower() != city_key for c in coords.keys()):
        coords[city_input.title()] = (float(lat), float(lon))
        save_city_data(coords)
        st.sidebar.success(f"✅ Added {city_input.title()} to saved cities!")

    city = city_input.title()
    st.subheader(f"🌍 Generating Forecast for {city}")
    st.markdown("---")

    st.info(f"Extracting data for {city} ({lat:.2f}, {lon:.2f}) ...")
    ts = extract_timeseries(ds, var, lat, lon)

    st.subheader("📈 Historical Time Series")
    st.line_chart(ts)

    with st.spinner("Training LSTM model and generating forecast..."):
        forecast, mae, rmse = lstm_forecast(ts, seq_length=30, steps=steps)

    st.subheader("🔮 Forecast Results")
    st.line_chart(pd.concat([ts[-100:], forecast]))

    trend = "increasing 📈" if forecast.iloc[-1] > ts.iloc[-1] else "decreasing 📉"
    st.success(f"**Detected Trend:** {trend}")
    st.write(f"**MAE:** {mae:.3f} | **RMSE:** {rmse:.3f}")

    st.markdown("### 🧠 Insight Summary")
    st.write(f"- The LSTM model predicts a {trend} trend in {var_descriptions[var].lower()} over the next {steps} days for {city}.")
    st.write(f"- Average forecast: **{forecast.mean():.2f}**")
    st.write(f"- Range: **{forecast.min():.2f} → {forecast.max():.2f}**")
    st.write(f"- Model shows stable short-term performance (RMSE ≈ {rmse:.2f}).")

    st.download_button(
        "💾 Download Forecast CSV",
        forecast.to_csv().encode(),
        file_name=f"{city}_{var}_forecast.csv",
        mime="text/csv"
    )

st.sidebar.markdown("---")
st.sidebar.info("**ClimateGuardX** — Explainable AI climate forecasting using LSTM.")
