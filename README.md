# Predicting Next-Day Stock Volatility Using Market Data and News Sentiment

[![CI](https://github.com/shreydevkar/CS506FinalProject/actions/workflows/ci.yml/badge.svg)](https://github.com/shreydevkar/CS506FinalProject/actions/workflows/ci.yml)

## 1. Project Description & Motivation

Financial markets are highly sensitive to both quantitative market signals and qualitative information such as news. Traders, hedge funds, and quantitative researchers attempt to predict volatility because it is directly tied to risk management, derivative pricing, and trading strategy performance.

The goal of this project is to build a predictive model that estimates **next-day realized volatility** of selected S&P 500 stocks using:

- Historical price and volume data  
- Technical indicators  
- News sentiment extracted from financial headlines  

This project follows the full data science lifecycle: data collection from financial APIs and news sources, data cleaning and preprocessing, feature extraction (technical and sentiment features), exploratory data visualization, and model training and evaluation.

The project aims to evaluate whether incorporating news sentiment improves volatility prediction compared to a clearly defined naive baseline model.



## 2. Target Variable Definition

Next-day realized volatility will be approximated using **squared daily log returns**, a commonly used proxy in financial econometrics when only daily price data is available.

### Daily Log Return

The daily log return is defined as:

rₜ = log(Pₜ / Pₜ₋₁)

Where:

- **Pₜ** = Adjusted closing price of the stock on day *t*  
- **Pₜ₋₁** = Adjusted closing price of the stock on the previous trading day  
- **rₜ** = Daily log return on day *t*  

Log returns are used instead of simple returns because they are time-additive and better suited for financial modeling.



### Realized Volatility Proxy

The realized volatility proxy is defined as:

RVₜ = rₜ²

Where:

- **RVₜ** = Realized volatility proxy on day *t*  
- Squaring removes the direction of returns and captures the magnitude of price fluctuations  

Squared returns are widely used as a volatility proxy because volatility measures variability, and larger price movements are emphasized.



### Prediction Target

The model’s prediction target is:

RVₜ₊₁

Where:

- **RVₜ₊₁** = Next-day realized volatility  

The objective is to predict tomorrow’s volatility using only information available up to time *t*, ensuring no forward-looking bias.



## 3. Project Goals

### Primary Goal

Successfully predict next-day realized volatility of selected S&P 500 stocks using historical market data and news sentiment features, achieving lower **Mean Squared Error (MSE)** than a clearly defined naive baseline.

### Specific and Measurable Objectives

- Collect at least 2 years of historical daily price and volume data  
- Compute realized volatility using squared log returns  
- Extract daily sentiment scores from financial news headlines  
- Engineer technical features including:
  - Rolling volatility  
  - Daily returns  
  - Volume change  
  - Moving averages  
  - RSI  
- Train and compare at least three models:
  - Linear Regression  
  - Random Forest  
  - Gradient Boosting (e.g., XGBoost)  
- Evaluate performance using:
  - Mean Squared Error (MSE)  
  - Mean Absolute Error (MAE)  
  - R² score  
- Compare results against a naive baseline model  



## 4. Baseline Model Definition

### 4.1 Primary naive baseline — Persistence model

The primary naive baseline is the **persistence model**, defined as:

RV̂ₜ₊₁ = RVₜ

Meaning: tomorrow's volatility is predicted to be equal to today's volatility. Volatility exhibits strong day-over-day persistence, so this is a non-trivial benchmark to beat.

All machine learning models are evaluated relative to this baseline, and improvement is reported as percentage reduction in MSE compared to the naive model.

### 4.2 Secondary econometric baseline — GARCH(1,1)

**Originally** this project compared only against the persistence baseline. The rationale has been strengthened to also include a **GARCH(1,1)** benchmark alongside persistence, for the reasons below.

**What GARCH(1,1) is:** the classical financial econometrics model for volatility (Bollerslev, 1986), parameterized as

σ²ₜ = ω + α·r²ₜ₋₁ + β·σ²ₜ₋₁

with three parameters (ω, α, β) fit on the returns series. It explicitly models the two stylized facts of financial volatility: **clustering** (past shocks predict current variance, captured by α) and **mean reversion** (variance drifts toward a long-run level, captured by β).

**Why persistence alone wasn't enough:**
1. Persistence is a *trivial* benchmark — it literally says "tomorrow = today" and has zero parameters. Beating it by 40–50% sounds impressive but is the expected baseline in any volatility paper.
2. The field's standard benchmark is GARCH, not persistence. A class project that only beats persistence could be dismissed as "didn't compare against the real benchmark."
3. GARCH is free to compute (3 params, seconds to fit) and widely available via the `arch` Python library.

**What adding GARCH revealed:** GARCH(1,1) beats the tuned ML ensemble on TSLA (the most volatile ticker) — 7.18e-07 vs. 8.03e-07. This is an honest and interesting finding: for heavy-tailed volatility regimes with pronounced mean reversion, the 3-parameter econometric prior still outperforms a Random Forest tuned on the same data. On AAPL and NKE the ML models win, but only by a few percent over GARCH rather than by the 40–50% they win over persistence.

**Takeaway:** the persistence baseline shows ML is doing something; the GARCH baseline shows *how much* of that something is just "rediscovering volatility clustering" versus contributing additional signal. Keeping both makes the comparison defensible.

Implementation lives in `src/garch_baseline.py` — rolling 1-step-ahead forecasts with a refit every 20 test days.



## 5. Data Collection Plan

### 5.1 Market Data

- **Source:** Yahoo Finance (via `yfinance` Python library)  
  https://finance.yahoo.com/  
  https://pypi.org/project/yfinance/  

- **Data Collected:**  
  - Open, High, Low, Close prices  
  - Adjusted Close prices  
  - Volume  
  - **Window used:** 2013-01-01 to 2018-01-01 (5 years of daily bars). The window was chosen to overlap with the Kaggle News Category Dataset's strong coverage period — HuffPost's article count dropped ~95% after 2018, making the 2013–2017 range the densest source of sentiment signal.

- **Method:**  
  API-based data retrieval via `yfinance`. Data is cached to `data/raw/{ticker}.csv` and reloaded on subsequent runs. Reproducible via `make run` or `python main.py --ticker <SYMBOL>`.



### 5.2 News Data

Two sources are wired via `src/news_sentiment.py`, switchable per call:

- **Primary: Kaggle News Category Dataset** (https://www.kaggle.com/datasets/rmisra/news-category-dataset)  
  210K HuffPost articles spanning 2012–2022. Downloaded once and cached at `data/raw/news_category_dataset.json` (gitignored, ~87 MB). Headlines are filtered per-ticker by a company-name regex (e.g. AAPL matches `Apple|iPhone|iPad|MacBook|Tim Cook|App Store`).  
  *Coverage in the 2013–2018 window*: AAPL ~499 days, TSLA ~109 days, NKE ~52 days of real signal (rest of trading days receive neutral 0s).

- **Supplemental: NewsAPI** (https://newsapi.org/) — fetched day-by-day via the `/v2/everything` endpoint. Free tier gives ~30 days of history, so it fills the *recent* end of a live deployment but does not cover the 2013–2018 training window. Yahoo Finance `Ticker.news` is also implemented as a no-key fallback.

- **Sentiment scoring:** VADER (NLTK) compound score on `title + short_description`. FinBERT was considered but not integrated — VADER is fast enough to score all headlines in one pass and the marginal benefit of FinBERT would be dominated by the coverage limitation.

- **Daily aggregates** per trading day:
  - `sentiment_mean`, `sentiment_std`, `news_count` (point-in-time)
  - `sentiment_mean_5d`, `news_count_5d` (5-day rolling — densifies the signal for tree models by giving non-zero values on days adjacent to a news event)



### 5.3 S&P 500 Stock List

- **Source:**  
  Wikipedia S&P 500 list  
  https://en.wikipedia.org/wiki/List_of_S%26P_500_companies  

This list will be used to select representative stocks for analysis.



### 5.4 Data Integration

Market data and sentiment data are merged on the daily date index inside `feature_engineering.add_features()`. Concretely:

- Sentiment columns are merged **before** `Target = RV.shift(-1)` is computed, so tomorrow's sentiment cannot leak into today's prediction.
- Days without any headline receive neutral sentiment (`sentiment_mean = 0`, `sentiment_std = 0`, `news_count = 0`) rather than being dropped — this preserves the full market time series.
- Train/test split is strictly chronological (first 80% train, last 20% test) to prevent lookahead bias.



## 6. Project Timeline (8 Weeks)

- ~~**Weeks 1–2:** Repository setup, market data collection pipeline, initial news data collection, and exploratory analysis~~ *(complete)*
- ~~**Weeks 3–4:** Data cleaning, technical indicator computation, sentiment scoring pipeline implementation, dataset merging, and preliminary visualizations~~ *(complete)*
- ~~**Weeks 5–6:** Baseline model implementation, training of Linear Regression, Random Forest, and Gradient Boosting models, and performance evaluation~~ *(complete)*
- **Week 7 (current):** Feature importance analysis, error analysis, and model refinement
- **Week 8:** Final visualizations, README completion, testing and GitHub workflow setup, and presentation recording  



## 7. Fallback Plan (Scope Control)

If incorporating news sentiment proves too complex within the project timeline, the scope will pivot to predicting next-day volatility using only market-derived technical features.

This ensures:

- The same dataset can be reused  
- The modeling pipeline remains intact  
- The project remains within the original volatility forecasting framework  

This avoids restarting the project mid-semester while preserving analytical depth.



## 8. Expected Deliverables

- Reproducible data collection scripts  
- Cleaned and processed datasets  
- Feature engineering pipeline  
- Clearly defined naive baseline model  
- Trained machine learning models  
- Evaluation metrics and baseline comparison  
- Feature importance analysis  
- Clear data visualizations  
- GitHub workflow for automated testing  
- Makefile for reproducibility  



## 9. Usage

### Install

```bash
make install              # installs from requirements.txt
```

First-time setup for sentiment features:

1. Download `News_Category_Dataset_v3.json` from https://www.kaggle.com/datasets/rmisra/news-category-dataset and place it at `data/raw/news_category_dataset.json` (the file is gitignored).
2. *(Optional)* For live NewsAPI fetches, create `.env` at the project root containing `NEWSAPI_KEY=<your key>`.

### Run

```bash
make run                                   # default ticker AAPL, market-only
python main.py --ticker AAPL               # same, explicit
python main.py --ticker TSLA --use-sentiment
python main.py --ticker AAPL --compare     # run both variants and print a side-by-side
```

The notebook `notebooks/preliminary_visualizations.ipynb` re-runs the pipeline for the showcase ticker and produces the comparison table, bar charts, feature-importance plots, and best-model overlay. Results CSVs are written to `notebooks/results.csv` (showcase) and `notebooks/results_all_tickers.csv` (cross-ticker).



## 10. Current Results

Test-set MSE on the 2013–2018 window (lower is better). Models include Linear Regression, **hyperparameter-tuned** Random Forest and XGBoost (4-fold expanding-window time-series CV grid search), and the classical **GARCH(1,1)** econometric benchmark. The persistence model (RV_{t+1} = RV_t) remains the primary baseline per §4.

### Best model per ticker (with sentiment features)

| Ticker | Best model | Best MSE | vs. Persistence | vs. GARCH(1,1) |
|---|---|---|---|---|
| **AAPL** | Random Forest (tuned) + sentiment | **1.08e-07** | **−47%** | **−7.6%** (beats GARCH) |
| **TSLA** | GARCH(1,1) | **7.18e-07** | **−46%** | — |
| **NKE**  | Random Forest (tuned) + sentiment | **6.02e-07** | **−50%** | **−3.4%** (beats GARCH) |

### Full results (all model × variant combinations) — see `notebooks/results_all_tickers.csv`

### Effect of adding news sentiment on tuned tree models

| Ticker | RF (no sent.) | RF (+ sent.) | Δ | XGB (no sent.) | XGB (+ sent.) | Δ |
|---|---|---|---|---|---|---|
| **AAPL** | 1.089e-07 | **1.078e-07** | **−1.0%** | 1.382e-07 | **1.110e-07** | **−19.7%** |
| **TSLA** | 8.032e-07 | **7.889e-07** | **−1.8%** | 1.122e-06 | **1.002e-06** | **−10.7%** |
| **NKE**  | 6.044e-07 | **6.022e-07** | **−0.4%** | 6.881e-07 | 7.038e-07 | +2.3% |

### Findings

- **All ML models beat the persistence baseline on all three tickers** (−44% to −50% MSE).
- **Sentiment helps tuned tree models consistently** (5 of 6 combinations), not just Linear Regression. The earlier finding that "sparse sentiment hurts trees" turned out to be a **hyperparameter issue, not a feature issue**: with properly regularized trees (`min_samples_leaf ≥ 10`, bounded depth, L2 on XGBoost), sentiment adds signal cleanly. The biggest single win is AAPL XGBoost, where sentiment cuts MSE by nearly 20%.
- **GARCH(1,1) is competitive** — it wins TSLA outright (the most volatile ticker) and is within 4% of the best ML model on AAPL/NKE. Including GARCH matters: volatility prediction's classical benchmark isn't persistence, and a ML project that beats only a naive baseline is less defensible than one that also contextualizes against GARCH.
- **Hyperparameter tuning was the largest single improvement** — TSLA RF alone went from 1.66e-05 (pathologically overfit) to 8.03e-07 (−95% MSE). The tuning grid (max_depth ∈ {5, 8, None}, min_samples_leaf ∈ {5, 10, 20}, plus n_estimators and max_features) was run per-ticker with 4-fold chronological CV on the training set only.
- **Honest limitation**: Kaggle/HuffPost headline coverage is not uniform across tickers (AAPL ~499 days, TSLA ~109, NKE ~52). A denser source (Finnhub, Reuters archive, or paid NewsAPI tier) would likely flip TSLA from GARCH-best to ML-best as well.



## 11. Alternative / Backup Project Idea: FitRec Gym Occupancy Prediction

As a backup, I am considering a project focused on **predicting gym occupancy at FitRec (BU’s fitness center)**. Instead of simply analyzing when the gym is busy, the project would be predictive, analytical, and decision-oriented.

### Proposed Approach

- Build a predictive model of gym occupancy using:
  - Temporal features (time of day, day of week)  
  - Environmental signals (weather)  
  - Academic signals (midterms, holidays)  

- Analyze the **relative importance of contextual features** to understand what drives gym attendance.

### Optional Extensions

1. **Forecasting Component:** Predict tomorrow’s gym busyness using time-series models.  
2. **Optimization Component:** Recommend the best time to go given user constraints, creating a personalized gym schedule.  
3. **Causal Insight Questions:**
   - Does bad weather increase indoor gym demand?  
   - Do midterm or finals weeks spike attendance?  

This backup project follows the full data science lifecycle: data collection, cleaning, feature engineering, visualization, modeling, evaluation, and optional decision-support analysis.
