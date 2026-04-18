# Predicting Next-Day Stock Volatility Using Market Data and News Sentiment



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

The primary naive baseline will be a **persistence model**, defined as:

RV̂ₜ₊₁ = RVₜ

Meaning: tomorrow’s volatility is predicted to be equal to today’s volatility.

This is a strong and widely used financial benchmark because volatility exhibits persistence over time.

All machine learning models will be evaluated relative to this baseline, and improvement will be reported as percentage reduction in MSE compared to the naive model.



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



## 10. Current Results (Week 7 Check-in)

Test-set MSE on the 2013–2018 window (lower is better, persistence baseline is the target to beat):

| Ticker | Baseline MSE | LR no sent. | LR + sent. | RF no sent. | XGB no sent. |
|---|---|---|---|---|---|
| **AAPL** | 2.03e-07 | 1.13e-07 | **1.12e-07** | 1.13e-07 | 1.09e-07 |
| **TSLA** | 1.33e-06 | 1.02e-06 | 1.04e-06 | 1.66e-05 *(overfit)* | 2.51e-06 |
| **NKE**  | 1.21e-06 | 6.02e-07 | 6.02e-07 | 6.59e-07 | 6.73e-07 |

Findings:
- All three ML models **beat the persistence baseline on AAPL and NKE**; Linear Regression beats it on TSLA.
- Sentiment gives AAPL Linear Regression a further **−0.79% MSE** on top of the market-only model. AAPL has by far the densest headline coverage (~499 days / ~40% of trading days), which explains why it's the only ticker where sentiment moves the needle meaningfully.
- Tree models (RF, XGBoost) do not benefit from sentiment in this dataset — the sparse, mostly-zero sentiment columns create splits that overfit the training period. The RF TSLA result is an outlier caused by heavy-tailed 2014–2015 volatility spikes in the training set.
- Honest limitation: the Kaggle dataset's historical depth is not uniform across tickers. AAPL dominates HuffPost's coverage; NKE barely appears. A denser source (Finnhub, Reuters archive, or paid NewsAPI tier) would raise TSLA and NKE into the same regime as AAPL.



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
