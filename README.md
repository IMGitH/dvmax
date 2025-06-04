
<p align="center">
  <img src="assets/logo.png" alt="Project Logo" width="200"/>
</p>

# Global Dividend ML Engine – Full Strategy Guide

This document is the **complete reference** for designing, validating, and operating the dividend-focused ML system. It covers:

- A robust end-to-end ML pipeline  
- Data engineering and validation practices  
- Practical storage & refresh architecture

---

## ✅ Project Goal

Build a machine learning engine that selects a basket of **non-U.S. dividend-paying stocks** that are:

- **High-yielding**
- **Low risk of dividend cuts**
- **Fundamentally stable**
- **Free of U.S. estate tax exposure**

---

## ✅ Core Principles

- Predict **12-month dividend cut probability**
- Focus on **non-U.S. equities** via **direct stocks or UCITS ETFs**
- Prioritize **stability, explainability**, and **investment practicality**
- Apply **rigorous data validation and historical consistency checks**

---

## ✅ End-to-End Pipeline

### Phase 1: Planning and Data Collection

- Select ~10–30 non-U.S. equities, rebalanced quarterly
- Primary source: **Financial Modeling Prep (FMP)**
- Optional: Yahoo Finance, World Bank, OECD
- Store all structured data in **Parquet format** under `/features_data/`

---

### Phase 2: Feature Engineering

Derived rolling/statistical features:

**Dividend & Valuation**
- Dividend yield  
- 3Y/5Y dividend CAGR  
- Payout ratio  
- Yield vs. 5Y median

**Financial**
- EPS & FCF CAGR  
- Net debt to EBITDA  
- EBIT interest coverage (capped & raw)

**Price-based**
- 6M / 12M return  
- Volatility, max drawdown (1Y)  
- Sector-relative return (6M)  
- SMA(50)/SMA(200) delta

**Contextual**
- Sector encoding  
- Country tag  
- Macro indicators (GDP, inflation, unemployment, etc.)

**Missing-Value Flags**
- Boolean `has_` columns added for key nullable metrics

---

### Phase 3: ML Modeling

- **Model**: `XGBoostClassifier`
- **Target**: Binary label – will dividend be cut within 12 months?
- **Training**: Walk-forward validation, with publication-lag shift to prevent lookahead
- **Evaluation**: ROC-AUC, F1, recall, calibration curves, and model stability across time

---

### Phase 4: Scoring and Optimization

1. **Filter stocks** by `cut_prob < 0.25`
2. **Score formula**:

   ```
   Final Score = (AdjYield × 0.5) + (DivGrowth × 0.3) + (Stability × 0.2)
   ```

3. **Component details**:
   - **AdjYield** = `RawYield × (1 − CutProbability)`
   - **DivGrowth** = 3–5Y CAGR, log-scaled/capped
   - **Stability** includes:
     - `(1 − CutProbability)`
     - Low drawdown/volatility
     - Conservative payout, debt
     - Dividend streaks

---

### Phase 5: Portfolio Construction & Rebalancing

- Equal or score-weighted
- Constraints:
  - ≤20% per country/sector
  - ≥5 countries/sectors
- Holding logic:
  - Only replace on score/risk deterioration
  - Min holding: 2–3 quarters

---

### Phase 6: Monitoring & Reporting

- Alerts on:
  - Score/risk drops  
  - Price crashes  
  - Yield compression  
- Annual metrics:
  - Realized vs. predicted yield  
  - Avoided dividend cuts  
  - Portfolio turnover and total return

---

### Phase 7: App Output

- Per-stock scores and sub-scores  
- Historical score and risk evolution  
- Highlight macro or feature drift warnings

---

## ✅ Data Management & Validation

### Key Enhancements

#### 🔒 Feature Validation

- **Hard bounds** on key features, e.g.:
  - `dividend_yield ∈ [0, 0.25]`
  - `pe_ratio ∈ [0, 300]`
  - `volatility ∈ [0, 1]`

- **Trend checks** on jumpy features, e.g.:
  - `dividend_yield`, `pfcf_ratio`, `eps_cagr_3y`
  - Flag if change >5× previous value

- Invalid rows are **quarantined** to `/features_data/_invalid/`

---

### Storage Hierarchy

```
features_data/
├── tickers_history/
│   └── AAPL.parquet
├── tickers_static/
│   └── static_ticker_info.parquet
├── macro_history/
│   └── france.parquet
├── _invalid/
│   └── TICKER_DATE.parquet
└── features_all_tickers_timeseries.parquet
```

All files saved as **compressed Parquet** with `compression='zstd'`.

---

### Macro Feature Handling

- **GDP/Capita YoY**, **Inflation**, **Unemployment**, **Exports**
- Uses World Bank API with `backfilled_year = as_of.year − 1`
- NaN checks performed before saving
- Features saved per country under `/macro_history/`

---

### Data Refresh Schedule

| Type          | Frequency | Notes                                   |
|---------------|-----------|-----------------------------------------|
| Price         | Weekly    | Needed for rolling return, drawdown     |
| Fundamentals  | Quarterly | Lag-aware (based on report availability)|
| Dividends     | Quarterly | Tracks declared vs. paid                |
| Macro         | Yearly    | Reference previous complete year        |

---

## ✅ Tech Stack Summary

| Layer         | Core Tools                              | Optional Tools                        |
|---------------|------------------------------------------|----------------------------------------|
| Modeling      | XGBoost, Polars, NumPy, Scikit-learn     | Quantum Optimizer (QUBO)               |
| Data Handling | PyArrow, Polars, DuckDB                  | Kafka (for real-time triggers)         |
| App / Infra   | Docker, GitHub Actions                   | FastAPI, Streamlit, BASE44, Lovable    |
| Storage       | Local Parquet, Cloudflare R2 / S3        | DuckDB SQL interface                   |

---

## ✅ Dev Progress Tracker

### Phase 1 — Core Engine
- ✅ FMP data fetcher  
- ✅ Polars feature pipeline  
- ✅ XGBoost training with walk-forward validation  
- ✅ Portfolio backtester

### Phase 2 — Infrastructure
- ✅ Docker + GitHub Actions  
- ✅ Dynamic feature range validation  
- ✅ Static info and macro ingestion  
- ⏳ Auto scheduler for retrain/report

### Phase 3 — Optimization & App
- ⏳ QUBO-based optimization  
- ⏳ Streamlit or FastAPI dashboard

### Phase 4 — Hosting
- ⏳ Migrate to R2/S3  
- ⏳ Optional inference API

---

Let this document serve as the **live blueprint** for your dividend stock ML engine — grounded in data validation, long-term stability, and institutional-grade hygiene.

*Last updated: 2025-06-03*
