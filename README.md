# Global Dividend ML Engine – Full Strategy Guide

This document serves as the **complete reference** for designing, storing, and operating the data pipeline and ML system for your global dividend stock selection model. It combines both:

- End-to-end ML strategy and modeling pipeline
- Practical data management plan for refresh, storage, and usage

---

## ✅ Project Goal

Build a machine learning engine that selects a basket of **non-U.S. dividend-paying stocks** with:

- **High net dividend yield**
- **Low risk of dividend cuts**
- **Stable financial and price profiles**
- **Minimal U.S. estate tax exposure**

---

## ✅ Core Principles

- Predict **probability of dividend cut within 12 months** (classification)
- Invest only in stocks with **low cut risk + attractive adjusted yield**
- Focus on **non-U.S. stocks**, via **direct holdings** or **UCITS ETFs**
- Emphasize **stability**, **explainability**, and **practical trading constraints**

---

## ✅ End-to-End Pipeline

### Phase 1: Planning and Data Collection

1. Define portfolio scope: 10–30 ex-U.S. stocks, quarterly rebalancing, low turnover.
2. Main data source: **Financial Modeling Prep (FMP)** (fundamentals, dividends, price history).
3. Others: Yahoo Finance (optional), Alpha Vantage (optional), macro (OECD, World Bank).
4. Store all structured data as **Parquet**.

---

### Phase 2: Feature Engineering

Use rolling and summary features (not raw time series):
- **Dividend**: yield, growth, payout ratios, streaks
- **Financial**: EPS/FCF growth, debt ratios, ROE, margin
- **Price-based**: 6M/12M return, drawdown, volatility, moving averages
- **Contextual**: sector encoding, currency risk, macro tags

---

### Phase 3: ML Modeling

- Model: **XGBoost Classifier**
- Target: Dividend cut in next 12M (binary)
- Cross-validation: walk-forward style
- Avoid lookahead bias by shifting features to match publication lag
- Output: **cut probability**, used for filtering and scoring

---

### Phase 4: Scoring and Optimization

- Score filtered safe stocks using:
  ```
  Final Score = (AdjYield × 0.5) + (DivGrowth × 0.3) + (Stability × 0.2)
  ```
- Rank and select top 10–30 stocks

---

### Phase 5: Portfolio Construction & Rebalancing

- Style: Equal-weight or weighted by score
- Constraints:
  - Max 20% per sector/country
  - Min 5 sectors/countries
- Stability logic:
  - Only replace stocks with significant score/risk gap
  - Apply minimum holding period (e.g., 2–3 quarters)
  - Avoid short-term price-driven churn

---

### Phase 6: Monitoring and Reporting

- Portfolio-level metrics:
  - Estimated vs realized yield
  - Stability score
  - Red flags: score drop, price crash, risk spike
- Report annually:
  - Total return, net yield, turnover rate, dividend cuts avoided

---

### Phase 7: App Output

- Scored stock list
- Estimated vs realized dividend yield (12M)
- Stability markers (drawdowns, risk change)
- Score attribution per stock (explainable AI)

---

## ✅ Data Management Strategy

### Why Weekly Price Data?

Even if you rebalance quarterly:
- Weekly prices are needed to calculate accurate **6M/12M returns**, drawdown, volatility
- Gaps in updates weaken rolling feature reliability

---

### Data Refresh Schedule

| Data Type        | Frequency  | Notes |
|------------------|------------|-------|
| Prices           | Weekly     | To compute rolling returns/features |
| Fundamentals     | Quarterly  | Sync with earnings release |
| Dividends        | Quarterly  | Use latest declared |
| Macro (optional) | Quarterly  | Lagged sources (OECD, World Bank) |

---

### File Storage: Parquet

Efficient, compressed, columnar format:

```
/data
  /prices/
    prices_raw.parquet
    prices_weekly_agg.parquet
  /fundamentals/
    fundamentals_q.parquet
  /dividends/
    dividend_history_q.parquet
  /macro/
    interest_rates.parquet
  /features/
    stock_features_YYYYQX.parquet
```

---

### Storage Locations

| Option         | Use Case                          |
|----------------|------------------------------------|
| Local disk     | Solo development                  |
| Cloud (S3, GCS)| Team use, deployment              |
| DuckDB         | SQL access to Parquet on disk     |

---

### Tools

- Pandas or Polars for pipelines
- DuckDB for hybrid SQL workflows
- pyarrow for Parquet IO

---

### Action Checklist

- [ ] Setup `/data/` structure in repo
- [ ] Weekly job to fetch and store raw prices
- [ ] Quarterly job to refresh fundamentals, dividends, macro
- [ ] All derived features stored in `/data/features/` as `.parquet`
- [ ] Use rolling windows to generate features weekly
- [ ] Document schema and logic in `/docs/` or notebooks

---

*Last updated: 2025-05-11*
