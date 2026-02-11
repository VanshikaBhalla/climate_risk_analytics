# 🌍 ClimateGuardX
## A Predictive & Explainable Climate Risk Analytics Dashboard

ClimateGuardX is a Python-based climate forecasting and explainable analytics platform designed to analyze long-term atmospheric data and generate localized climate predictions. The system integrates statistical, machine learning, and deep learning approaches into a unified pipeline and presents results through an interactive Streamlit dashboard.

The project focuses on temperature and precipitation forecasting using historical climate data and provides both accurate predictions and interpretable insights for researchers, policymakers, and environmental analysts.

---

<img width="735" height="378" alt="image" src="https://github.com/user-attachments/assets/0d06cad3-b51f-479f-bd16-0066f20ebb3e" />


## 📌 Overview

Accurate climate forecasting is critical for agriculture, disaster preparedness, and environmental planning. Traditional statistical models often fail to capture nonlinear dependencies and long-term temporal relationships in climate data.

ClimateGuardX addresses this by combining:

- Statistical Time-Series Modeling (ARIMA)
- Deep Learning (LSTM)
- Ensemble Learning (Random Forest)
- Explainable AI (SHAP)
- Interactive Visualization (Streamlit Dashboard)

The system processes IMDAA (Indian Monsoon Data Assimilation and Analysis) NetCDF datasets spanning 1990–2020 to generate localized climate forecasts and anomaly analyses.

---

## 🚀 Features

- Localized climate prediction for Indian cities
- Time-series forecasting using multiple models
- Explainable predictions using SHAP feature attribution
- Comparative model evaluation (MAE, RMSE, SMAPE)
- Temperature anomaly detection
- Interactive Streamlit dashboard
- Real-time access via Ngrok tunneling
- Modular and scalable architecture

---

## 🧠 Models Compared

### 1. ARIMA (AutoRegressive Integrated Moving Average)
- Serves as a statistical baseline model
- Captures linear temporal dependencies
- Suitable for stationary climate patterns

### 2. LSTM (Long Short-Term Memory Network)
- Deep learning model for sequential data
- Captures nonlinear and long-range dependencies
- Achieved the lowest prediction error
- Final deployed model in dashboard

### 3. Random Forest Regressor + SHAP
- Ensemble learning approach
- Uses lag-based features for prediction
- Provides interpretability via SHAP explanations
- Identifies influential climate factors

---

## 📊 Dataset

Dataset Used: [IMDAA Reanalysis Dataset](https://www.kaggle.com/datasets/maslab/bharatbench)

- Time range: 1990 – 2020
- Format: NetCDF
- Daily temporal resolution
- Geographic coverage: Indian region

Key variables:
- HGT_prl – Geopotential height
- TMP_prl – Atmospheric temperature
- TMP_2m – Surface temperature
- APCP_sfc – Surface precipitation

---

## ⚙️ System Architecture

The workflow consists of five main modules:

1. Data Acquisition & Preprocessing
   - NetCDF loading using xarray
   - Missing value handling
   - Unit conversion (Kelvin → Celsius)
   - Rolling mean computation
   - Anomaly detection

2. Feature Engineering
   - Lag feature generation (lag_1 to lag_7)

3. Model Training (on selected locations)
   - ARIMA
   - LSTM
   - Random Forest

4. Evaluation & Explainability
   - MAE
   - RMSE
   - SMAPE
   - SHAP feature importance

5. Visualization & Deployment
   - Streamlit dashboard
   - Ngrok-based public access

---

## 📈 Results Summary

| Model | MAE | RMSE |
|------|------|------|
| ARIMA (5,1,0) | 16.531 | 20.352 |
| LSTM | **3.303** | **4.193** |
| Random Forest + SHAP | 3.646 | 4.634 |

Key findings:

- LSTM achieved the best predictive performance.
- Random Forest provided strong interpretability.
- ARIMA served as a reliable statistical baseline.
- Forecasts indicate a slight long-term warming trend across cities.

---

## 🖥️ Dashboard

The Streamlit dashboard allows users to:

- Select cities
- Visualize historical climate trends
- View anomaly plots
- Generate LSTM-based forecasts
- Explore prediction results interactively

---

## 🧰 Tech Stack

### Language
- Python 3.12

### Libraries
- xarray
- numpy
- pandas
- statsmodels
- tensorflow.keras
- scikit-learn
- matplotlib
- shap
- streamlit
- pyngrok
- tqdm

---
