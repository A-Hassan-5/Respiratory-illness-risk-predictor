# 🫁 Respiratory Illness Risk Predictor

> A machine learning–powered dashboard that predicts respiratory illness risk from air quality and weather data — supporting public health awareness and early smog-related risk warnings across the United States.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running the App](#running-the-app)
- [Dependencies](#dependencies)

---

## 🔍 Project Overview

This project predicts respiratory illness risk using a combination of **air pollution**, **weather**, and **seasonal epidemiological features**. It leverages machine learning models to estimate whether environmental conditions pose a **Low**, **Medium**, or **High** risk to respiratory health, and provides an interactive dashboard for:

- Real-time risk prediction
- Short-term forecasting
- Spatial analysis across US states

The system is designed to support **public health awareness**, **early warning systems**, and **data-driven insights** into smog-related respiratory risks.

---

## 🎯 Objectives

- Predict respiratory illness risk based on smog and weather data
- Compare multiple machine learning models
- Select and deploy the best-performing model
- Visualize risk trends across time and US states
- Provide short-term forecasting insights

---

## 📊 Dataset Description

| Property | Details |
|---|---|
| **Records** | ~213,000 rows |
| **Time Span** | 2015 – 2025 |
| **Geographic Scope** | United States (state level) |

### Key Features Used

**Air Quality Indicators** — pollutant concentration levels and AQI metrics used as primary model inputs.

> **Note:** State names are retained for analysis and visualization but are **not used as model inputs** to improve generalization across geographic regions.

---

## 🗂️ Project Structure

```
Respiratory-illness-risk-predictor/
├── app.py
├── rf_respiratory_risk.pkl
├── scaler.pkl
├── imputed_daily_AQ_2015_2025.csv
├── model.ipynb
├── requirements.txt
└── venv/
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.9+
- `pip` package manager

### A) Create and Activate a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### B) Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the App

```bash
streamlit run app.py
```

The app will be available at: **http://localhost:8501**

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Interactive web dashboard |
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `scikit-learn` | Machine learning models |
| `joblib` | Model serialization |
| `plotly` | Interactive visualizations |
| `matplotlib` | Static plotting |
| `seaborn` | Statistical data visualization |
| `statsmodels` | Time series & forecasting |

Install all dependencies at once:

```bash
pip install streamlit pandas numpy scikit-learn joblib plotly matplotlib seaborn statsmodels
```
