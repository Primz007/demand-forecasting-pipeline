# Demand Forecasting System

An end-to-end time-series forecasting pipeline that predicts weekly product demand using SARIMAX and Prophet, built on real Walmart retail sales data - including seasonal decomposition, stationarity testing, and an honest comparison between two forecasting approaches.

## Problem Statement

Businesses in retail, FMCG, and logistics rely on accurate demand forecasts to plan inventory, avoid stockouts/overstocking, and optimize revenue. This project builds a forecasting pipeline on a single product-store series, evaluates it rigorously against a full seasonal cycle, and compares two different modeling approaches to understand their real trade-offs.

## Dataset

**Walmart Recruiting - Store Sales Forecasting** (Kaggle) - weekly sales data across 45 stores and 81 departments, Feb 2010 to Oct 2012 (143 weeks), no missing values.

For this project, one series was selected: **Store 1, Department 1**, chosen for having a complete, gap-free history and a clear, consistent yearly seasonal pattern - better suited to demonstrating decomposition and evaluation than a noisy or flat series.

## Approach

1. **EDA** - visualized the series, identified strong yearly seasonality and a flat overall trend
2. **Seasonal decomposition** - split into trend, seasonal, and residual components (period=52). Residuals showed sharp, unexplained spikes around April each year
3. **Stationarity testing (ADF)** - original series was non-stationary; one round of seasonal differencing (lag=52) achieved stationarity (p ≈ 1.2e-14)
4. **ACF/PACF analysis** - PACF showed a sharp cutoff after lag 2 (→ AR order p=2); ACF showed no MA-cutoff signature (→ q=0)
5. **SARIMAX** - compared SARIMAX(2,0,0)(1,1,1,52) against nearby parameter combinations via AIC; the ACF/PACF-derived parameters won outright
6. **Prophet** - fit with US holiday-awareness enabled, specifically to test whether explicit holiday handling could address a weakness found in SARIMAX (see Results)
7. **Evaluation** - both models evaluated on a full 52-week held-out test set (chosen deliberately to include seasonal peaks, not just quiet baseline weeks) using MAPE and RMSE

## Results

| Model | MAPE | RMSE |
|---|---|---|
| SARIMAX(2,0,0)(1,1,1,52) | 15.75% | 8053.25 |
| Prophet (holiday-aware) | 23.48% | 7561.68 |

**Key finding:** SARIMAX achieved lower average error but was fragile to a single calendar-shifting holiday event. The series showed an unexplained spike each April during decomposition - later confirmed during evaluation to be tied to **Easter**, whose date shifts year to year and isn't captured by a fixed 52-week seasonal model. SARIMAX correctly forecasted two of three major seasonal peaks but significantly underforecast the Easter-related spike, inflating its RMSE.

Prophet, with built-in holiday-awareness, correctly identified the *timing* of all three peaks - including the Easter one that SARIMAX missed - but consistently **underestimated peak magnitude**. This is likely due to limited training history (~2 years, meaning Prophet saw this holiday pattern only once or twice). Switching Prophet to multiplicative seasonality mode had minimal effect, suggesting the underestimation is driven more by insufficient training examples than by seasonality-mode misspecification.

**This reflects a genuine trade-off, not one model being objectively better:** SARIMAX is highly accurate under stable, repeating conditions but fragile to irregular calendar effects. Prophet trades some baseline precision for robustness to holidays - a trade-off that would likely shift further in Prophet's favor with more historical training data.

## Tech Stack

- **Language:** Python (Google Colab)
- **Data handling:** Pandas, NumPy
- **Modeling:** Statsmodels (SARIMAX), Prophet
- **Evaluation:** Scikit-learn (MAPE, RMSE)
- **Visualization:** Matplotlib

## Limitations

- Built on a single series (Store 1, Dept 1) - not yet validated across multiple stores/departments
- ~2 years of training data limits how well holiday effects can be learned, particularly for Prophet


## Why This Project

Demand forecasting is a widely-used, high-impact application of time-series ML in industry. Beyond fitting models, this project focuses on evaluating honestly - testing on a window that includes seasonal peaks rather than an easy stretch, diagnosing *why* each model succeeds or fails where it does, and drawing a clear, defensible comparison rather than declaring an arbitrary winner.
