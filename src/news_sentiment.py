"""News collection and sentiment scoring for volatility prediction.

Primary source: NewsAPI /v2/everything (free tier: ~30 days history, 100 req/day).
Fallback source: yfinance Ticker.news (recent headlines, no key).
Cache: data/raw/news_headlines_{ticker}.csv — fetched once, reused thereafter.

Sentiment: VADER compound score per headline, aggregated per trading day into
(sentiment_mean, sentiment_std, news_count).
"""
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer

load_dotenv()

NEWSAPI_URL = "https://newsapi.org/v2/everything"
CACHE_DIR = "data/raw"

TICKER_QUERY = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "NKE": "Nike",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "META": "Meta OR Facebook",
    "NVDA": "Nvidia",
}


def _cache_path(ticker):
    return os.path.join(CACHE_DIR, f"news_headlines_{ticker}.csv")


def _newsapi_day_request(query, day, per_day_cap, api_key):
    """One NewsAPI request scoped to a single calendar day. Returns list of article dicts."""
    params = {
        "q": query,
        "from": day.strftime("%Y-%m-%d"),
        "to": day.strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": min(per_day_cap, 100),
        "apiKey": api_key,
    }
    resp = requests.get(NEWSAPI_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("articles", [])


def fetch_news_newsapi(ticker, days_back=30, per_day_cap=30, api_key=None, pause=0.2):
    """Fetch headlines from NewsAPI, one calendar day at a time.

    Why per-day: a single bulk request returns the 100 most-recent articles which collapse
    onto the latest 1-3 days for busy tickers. Querying day-by-day spreads coverage across
    the full 30-day window. Uses days_back requests (~30) out of the 100/day free quota.
    """
    api_key = api_key or os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise RuntimeError("NEWSAPI_KEY missing — set it in .env or pass api_key=")

    query = TICKER_QUERY.get(ticker, ticker)
    today = datetime.utcnow().date()

    rows = []
    for offset in range(days_back):
        day = today - timedelta(days=offset)
        try:
            articles = _newsapi_day_request(query, day, per_day_cap, api_key)
        except requests.HTTPError as e:
            # Stop on quota/rate issues rather than burning more requests
            print(f"[news_sentiment] NewsAPI HTTP error on {day}: {e}; stopping early")
            break
        for a in articles:
            published = a.get("publishedAt", "")
            if not published:
                continue
            ts = pd.to_datetime(published, utc=True).tz_convert(None).normalize()
            rows.append({
                "date": ts,
                "title": (a.get("title") or "").strip(),
                "description": (a.get("description") or "").strip(),
                "source": (a.get("source") or {}).get("name", ""),
            })
        time.sleep(pause)
    return pd.DataFrame(rows)


def fetch_news_yfinance(ticker):
    """Fallback: yfinance provides recent headlines without a key. Limited depth (~10-20 items)."""
    import yfinance as yf
    items = yf.Ticker(ticker).news or []
    rows = []
    for item in items:
        content = item.get("content", item)
        pub = content.get("pubDate") or content.get("providerPublishTime")
        if not pub:
            continue
        if isinstance(pub, (int, float)):
            ts = pd.to_datetime(pub, unit="s", utc=True)
        else:
            ts = pd.to_datetime(pub, utc=True)
        rows.append({
            "date": ts.tz_convert(None).normalize(),
            "title": content.get("title", "") or "",
            "description": content.get("summary", "") or "",
            "source": (content.get("provider") or {}).get("displayName", "") if isinstance(content.get("provider"), dict) else "",
        })
    return pd.DataFrame(rows)


def load_news_cache(ticker):
    """Load cached headlines if they exist."""
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def save_news_cache(df, ticker):
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(_cache_path(ticker), index=False)


def get_or_fetch_headlines(ticker, use_cache=True, days_back=30):
    """Return headlines DataFrame. Reads cache if present; else fetches and caches.

    Tries NewsAPI first, falls back to yfinance if that fails.
    """
    if use_cache:
        cached = load_news_cache(ticker)
        if cached is not None and len(cached) > 0:
            return cached

    try:
        df = fetch_news_newsapi(ticker, days_back=days_back)
        if len(df) == 0:
            raise RuntimeError("NewsAPI returned 0 articles")
    except Exception as e:
        print(f"[news_sentiment] NewsAPI failed ({e}); falling back to yfinance")
        df = fetch_news_yfinance(ticker)

    save_news_cache(df, ticker)
    return df


_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        try:
            _analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            import nltk
            nltk.download("vader_lexicon", quiet=True)
            _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def compute_vader_sentiment(headlines_df):
    """Add VADER compound score column to headlines DataFrame.

    Scores title+description concatenated (more signal than title alone).
    Compound range: [-1 (most negative), +1 (most positive)].
    """
    if len(headlines_df) == 0:
        return headlines_df.assign(compound=pd.Series(dtype=float))
    sia = _get_analyzer()
    text = (headlines_df["title"].fillna("") + ". " + headlines_df["description"].fillna("")).str.strip()
    scores = text.apply(lambda t: sia.polarity_scores(t)["compound"] if t else 0.0)
    return headlines_df.assign(compound=scores)


def aggregate_daily_sentiment(scored_df):
    """Collapse per-headline scores to per-date features.

    Returns DataFrame indexed by date with sentiment_mean, sentiment_std, news_count.
    """
    if len(scored_df) == 0:
        return pd.DataFrame(columns=["sentiment_mean", "sentiment_std", "news_count"])

    agg = scored_df.groupby("date")["compound"].agg(
        sentiment_mean="mean",
        sentiment_std="std",
        news_count="count",
    )
    agg["sentiment_std"] = agg["sentiment_std"].fillna(0.0)
    return agg


def build_sentiment_features(ticker, use_cache=True, days_back=30):
    """End-to-end: fetch/load headlines → score → aggregate daily. Returns daily sentiment DF."""
    headlines = get_or_fetch_headlines(ticker, use_cache=use_cache, days_back=days_back)
    scored = compute_vader_sentiment(headlines)
    return aggregate_daily_sentiment(scored)
