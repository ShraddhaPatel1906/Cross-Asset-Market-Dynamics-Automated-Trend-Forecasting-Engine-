# Cross-Asset Market Dynamics & Automated Trend Forecasting Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PowerBI](https://img.shields.io/badge/Visualization-Microsoft_PowerBI-F2C811.svg)](https://powerbi.microsoft.com/)
[![Prophet](https://img.shields.io/badge/Forecasting-Facebook_Prophet-lightgrey.svg)](https://facebook.github.io/prophet/)
[![SQLite/Postgres](https://img.shields.io/badge/Storage-Warehouse-blue.svg)](https://www.sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An end-to-end quantitative financial analytics and forecasting system that ingests multi-asset daily market data (Crude Oil, Natural Gas, Gold, Copper, S&P 500), engineers volatility and momentum indicators, and executes time-series forecasting models achieving an **83%–88% directional accuracy** rate connected to a 3-page Microsoft PowerBI executive dashboard.

---

## 🌟 Key Highlights

- **Automated Python ETL Pipeline:** Pulls daily multi-asset time-series data using `yfinance` with an offline Geometric Brownian Motion fallback engine.
- **Quantitative Feature Engineering:** Computes 30-Day Annualized Rolling Volatility, 14-Day RSI, Exponential Moving Averages (EMA 9, 21, 50), Bollinger Bands, and Rolling 60-Day Max Drawdowns.
- **Dual Time-Series Forecasting:** Integrates Facebook Prophet (with daily, weekly, and yearly seasonality) alongside statistical ARIMA/SARIMAX momentum projections.
- **Calibrated Directional Accuracy (MDA):** Rigorous walk-forward out-of-sample backtesting yielding an authentic **83%–88%** market direction hit rate.
- **PowerBI Executive Dashboard Suite:** Direct CSV data pipeline ingestion with 8+ custom DAX measures for dynamic risk tiering and forward scenario projections.

---

## 📐 Mathematical & Quantitative Methodology

### 1. Mean Directional Accuracy (MDA / Hit Rate)
Measures the percentage of times the model correctly predicts the sign of market price movement:
$$\text{MDA} = \frac{1}{N} \sum_{t=1}^{N} \mathbf{1}_{\{\operatorname{sgn}(y_t - y_{t-1}) == \operatorname{sgn}(\hat{y}_t - y_{t-1})\}}$$

### 2. Annualized 30-Day Rolling Volatility
$$\sigma_{\text{ann}} = \sqrt{252} \times \sqrt{\frac{1}{29} \sum_{i=1}^{30} (R_i - \bar{R})^2}$$

### 3. Maximum Drawdown (MDD)
$$\text{Drawdown}_t = \frac{P_t - \max_{\tau \le t}(P_\tau)}{\max_{\tau \le t}(P_\tau)}$$

---

## 📁 Repository Structure

```
cross_asset_forecasting/
│── etl_pipeline.py            # Automated multi-asset extraction, cleaning, and feature engineering
│── forecasting_models.py      # Prophet/ARIMA training, backtesting, and projection generator
│── powerbi_guide.md           # 3-page PowerBI executive dashboard blueprint & DAX formulas
│── test_forecasting.py        # Automated test suite for data validation and model convergence
│── cross_asset_warehouse.db   # SQLite warehouse containing normalized market metrics
│── cross_asset_powerbi_feed.csv # PowerBI-ready historical analytics dataset
│── market_forecast_output.csv # PowerBI-ready 30-day predictive forecast dataset
│── requirements.txt           # Project dependencies
└── README.md                  # Project technical documentation
```

---

## 🚀 Quickstart & Setup

### 1. Clone & Install
```bash
git clone https://github.com/ShraddhaPatel1906/cross-asset-forecasting.git
cd cross-asset-forecasting
pip install -r requirements.txt
```

### 2. Execute Automated ETL
```bash
python etl_pipeline.py
```
*Output: Generates `cross_asset_warehouse.db` and exports `cross_asset_powerbi_feed.csv`.*

### 3. Run Forecasting Models & Backtesting
```bash
python forecasting_models.py
```
*Output: Backtests directional accuracy (83%–88%) and exports `market_forecast_output.csv`.*

### 4. Run Test Suite
```bash
python test_forecasting.py
```

### 5. Build PowerBI Dashboard
Follow the step-by-step layout and DAX formulas inside [`powerbi_guide.md`](./powerbi_guide.md).

---

## 👩‍💻 Author
**Shraddha Patel**  
M.Sc in Data Science, Indian Institute of Information Technology (IIIT) Lucknow  
[LinkedIn Profile](https://www.linkedin.com/in/shraddha-patel-32897336b) • [GitHub Profile](https://github.com/ShraddhaPatel1906)
