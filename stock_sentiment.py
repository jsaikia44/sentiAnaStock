"""
NSE Stock News Sentiment Analyzer
---------------------------------
Fetches recent news for NSE (India) stocks, scores headline sentiment,
and produces a simple bullish / bearish / neutral signal.

News sources:
  - Google News India (RSS)  -> good coverage of Indian business news
  - Yahoo Finance            -> extra headlines when available

Install:
    pip install yfinance pandas nltk feedparser
    # optional, finance-specific model (more accurate, bigger download):
    pip install transformers torch

Usage:
    python stock_sentiment.py RELIANCE
    python stock_sentiment.py TCS INFY HDFCBANK
    python stock_sentiment.py SBIN --model finbert
    python stock_sentiment.py ^NSEI           # Nifty 50 index

The ".NS" suffix is added automatically (RELIANCE -> RELIANCE.NS).

DISCLAIMER: News sentiment is a weak, noisy signal and does NOT reliably
predict price moves. For research/learning only. Not financial advice.
"""

import argparse
import math
import urllib.parse
from datetime import datetime, timezone

import feedparser
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# 1. Ticker helpers
# ---------------------------------------------------------------------------
def to_nse_symbol(ticker: str) -> str:
    """RELIANCE -> RELIANCE.NS (leave indices like ^NSEI and .NS/.BO alone)."""
    t = ticker.upper().strip()
    if t.startswith("^") or t.endswith(".NS") or t.endswith(".BO"):
        return t
    return t + ".NS"


def company_name(symbol: str) -> str:
    """Look up the company name for a better news search."""
    try:
        info = yf.Ticker(symbol).info
        name = info.get("shortName") or info.get("longName")
        if name:
            # strip common suffixes so the search isn't too narrow
            for s in [" Limited", " Ltd.", " Ltd", " LIMITED", " LTD"]:
                name = name.replace(s, "")
            return name.strip()
    except Exception:
        pass
    return symbol.replace(".NS", "").replace(".BO", "")


# ---------------------------------------------------------------------------
# 2. Fetch news
# ---------------------------------------------------------------------------
def fetch_google_news(query: str, days: int = 7) -> list:
    """Indian business headlines from Google News RSS."""
    q = urllib.parse.quote(f"{query} share price when:{days}d")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    rows = []
    for e in feed.entries:
        title = e.get("title", "")
        # Google appends " - Source Name"; separate it out
        source = ""
        if " - " in title:
            title, source = title.rsplit(" - ", 1)
        published = None
        if e.get("published_parsed"):
            published = pd.Timestamp(datetime(*e.published_parsed[:6], tzinfo=timezone.utc))
        rows.append({"title": title.strip(), "summary": "", "source": source,
                     "published": published})
    return rows


def fetch_yahoo_news(symbol: str) -> list:
    """Headlines from Yahoo Finance (often sparse for Indian stocks)."""
    try:
        raw = yf.Ticker(symbol).news or []
    except Exception:
        return []
    rows = []
    for item in raw:
        content = item.get("content", item)
        title = content.get("title", "")
        published = None
        if "pubDate" in content:
            published = pd.to_datetime(content["pubDate"], utc=True)
        elif "providerPublishTime" in item:
            published = pd.to_datetime(item["providerPublishTime"], unit="s", utc=True)
        if title:
            rows.append({"title": title, "summary": content.get("summary", "") or "",
                         "source": "Yahoo", "published": published})
    return rows


def fetch_news(symbol: str, days: int = 7) -> pd.DataFrame:
    name = company_name(symbol)
    rows = fetch_google_news(name, days) + fetch_yahoo_news(symbol)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset="title")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    df = df[df["published"].isna() | (df["published"] >= cutoff)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Sentiment models
# ---------------------------------------------------------------------------
class VaderScorer:
    """Fast general-purpose sentiment, score in [-1, 1]."""

    # Add finance/Indian-market words VADER doesn't know
    EXTRA_LEXICON = {
        "bullish": 2.5, "bearish": -2.5, "upgrade": 2.0, "downgrade": -2.0,
        "outperform": 2.0, "underperform": -2.0, "beat": 1.5, "beats": 1.5,
        "miss": -1.5, "misses": -1.5, "surge": 2.0, "surges": 2.0, "soars": 2.5,
        "rally": 2.0, "rallies": 2.0, "plunge": -2.5, "plunges": -2.5,
        "slump": -2.0, "slumps": -2.0, "tanks": -2.5, "crash": -3.0,
        "profit": 1.0, "loss": -1.5, "losses": -1.5, "dividend": 1.0,
        "buyback": 1.5, "bonus": 1.0, "penalty": -2.0, "sebi probe": -2.5,
        "fraud": -3.0, "default": -2.5, "pledge": -1.0, "order win": 2.0,
        "52-week high": 2.0, "52-week low": -2.0, "upper circuit": 2.5,
        "lower circuit": -2.5, "target price raised": 2.0, "target cut": -2.0,
    }

    def __init__(self):
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        self.sia = SentimentIntensityAnalyzer()
        self.sia.lexicon.update(self.EXTRA_LEXICON)

    def score(self, texts):
        return [self.sia.polarity_scores(t)["compound"] for t in texts]


class FinBertScorer:
    """FinBERT, a model fine-tuned on financial text. Score in [-1, 1]."""

    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)

    def score(self, texts):
        results = self.pipe(list(texts), truncation=True)
        out = []
        for res in results:
            p = {r["label"].lower(): r["score"] for r in res}
            out.append(p.get("positive", 0) - p.get("negative", 0))
        return out


def get_scorer(name: str):
    if name == "finbert":
        try:
            return FinBertScorer()
        except Exception as e:
            print(f"[!] FinBERT unavailable ({e}); falling back to VADER.")
    return VaderScorer()


# ---------------------------------------------------------------------------
# 4. Aggregate into a signal
# ---------------------------------------------------------------------------
def recency_weight(published, half_life_hours=24.0):
    """Newer news counts more: weight halves every `half_life_hours`."""
    if published is None or pd.isna(published):
        return 0.5
    age_h = (datetime.now(timezone.utc) - published).total_seconds() / 3600
    return math.pow(0.5, max(age_h, 0) / half_life_hours)


def analyze(ticker: str, scorer, threshold=0.15, days=7) -> dict:
    symbol = to_nse_symbol(ticker)
    df = fetch_news(symbol, days)
    if df.empty:
        return {"ticker": symbol, "signal": "NO DATA", "score": None}

    text = (df["title"] + ". " + df["summary"]).str.strip(" .")
    df["sentiment"] = scorer.score(text)
    df["weight"] = df["published"].apply(recency_weight)
    weighted = (df["sentiment"] * df["weight"]).sum() / df["weight"].sum()

    if weighted > threshold:
        signal = "UP (bullish)"
    elif weighted < -threshold:
        signal = "DOWN (bearish)"
    else:
        signal = "NEUTRAL"

    hist = yf.Ticker(symbol).history(period="5d")
    last_price = change_5d = None
    if len(hist) >= 2:
        last_price = hist["Close"].iloc[-1]
        change_5d = (last_price / hist["Close"].iloc[0] - 1) * 100

    return {
        "ticker": symbol,
        "signal": signal,
        "score": weighted,
        "n_articles": len(df),
        "pct_positive": (df["sentiment"] > 0.05).mean() * 100,
        "pct_negative": (df["sentiment"] < -0.05).mean() * 100,
        "last_price": last_price,
        "price_change_5d": change_5d,
        "news": df.sort_values("published", ascending=False, na_position="last"),
    }


# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------
IST = "Asia/Kolkata"


def print_report(r: dict):
    print("=" * 78)
    print(f" {r['ticker']}")
    print("=" * 78)
    if r["score"] is None:
        print(" No recent news found. Check the symbol (e.g. HDFCBANK, not HDFC Bank).\n")
        return

    print(f" Articles analyzed  : {r['n_articles']}")
    print(f" Weighted sentiment : {r['score']:+.3f}  (range -1 to +1)")
    print(f" Positive / Negative: {r['pct_positive']:.0f}% / {r['pct_negative']:.0f}%")
    if r["last_price"] is not None:
        print(f" Last close         : \u20b9{r['last_price']:,.2f}  "
              f"({r['price_change_5d']:+.2f}% over 5 days)")
    print(f" SIGNAL             : {r['signal']}")
    print("-" * 78)

    for _, row in r["news"].head(12).iterrows():
        when = (row["published"].tz_convert(IST).strftime("%d-%b %H:%M")
                if pd.notna(row["published"]) else "   unknown  ")
        src = f" ({row['source']})" if row["source"] else ""
        print(f" [{row['sentiment']:+.2f}] {when}  {row['title'][:60]}{src[:20]}")
    print()


def main():
    parser = argparse.ArgumentParser(description="NSE stock news sentiment analysis")
    parser.add_argument("tickers", nargs="+", help="NSE symbols, e.g. RELIANCE TCS INFY")
    parser.add_argument("--model", choices=["vader", "finbert"], default="vader")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Score needed to call UP/DOWN (default 0.15)")
    parser.add_argument("--days", type=int, default=7, help="News lookback in days")
    args = parser.parse_args()

    scorer = get_scorer(args.model)
    summary = []
    for t in args.tickers:
        r = analyze(t, scorer, args.threshold, args.days)
        print_report(r)
        summary.append({"ticker": r["ticker"], "signal": r["signal"],
                        "score": None if r["score"] is None else round(r["score"], 3)})

    if len(summary) > 1:
        print(pd.DataFrame(summary).to_string(index=False))

    print("\nReminder: sentiment is not a reliable price predictor. Not financial advice.")


if __name__ == "__main__":
    main()
