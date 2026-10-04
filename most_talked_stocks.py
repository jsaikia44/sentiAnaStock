"""
Most-Talked-About NSE Stocks Scanner
------------------------------------
Ranks stocks by how much "buzz" they have right now, using two signals:

  1. News buzz    -> number of headlines in the last N days (Google News India)
  2. Trading buzz -> today's volume vs. its 20-day average (volume spike)

It then scores the sentiment of each stock's headlines so you can see
whether the talk is positive or negative.

Install:
    pip install yfinance pandas nltk feedparser

Usage:
    python most_talked_stocks.py                     # scan Nifty 50, last 2 days
    python most_talked_stocks.py --days 1 --top 15
    python most_talked_stocks.py --file my_list.txt  # one NSE symbol per line

Note: the Nifty 50 list below changes a few times a year. Check it against
nseindia.com, or pass your own list with --file.

DISCLAIMER: "Most talked about" is not the same as "will go up".
Heavy buzz often comes with high volatility. Not financial advice.
"""

import argparse
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import feedparser
import nltk
import pandas as pd
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Symbol -> name used for the news search
NIFTY50 = {
    "ADANIENT": "Adani Enterprises", "ADANIPORTS": "Adani Ports",
    "APOLLOHOSP": "Apollo Hospitals", "ASIANPAINT": "Asian Paints",
    "AXISBANK": "Axis Bank", "BAJAJ-AUTO": "Bajaj Auto",
    "BAJFINANCE": "Bajaj Finance", "BAJAJFINSV": "Bajaj Finserv",
    "BEL": "Bharat Electronics", "BHARTIARTL": "Bharti Airtel",
    "CIPLA": "Cipla", "COALINDIA": "Coal India", "DRREDDY": "Dr Reddy's",
    "EICHERMOT": "Eicher Motors", "ETERNAL": "Eternal Zomato",
    "GRASIM": "Grasim", "HCLTECH": "HCL Tech", "HDFCBANK": "HDFC Bank",
    "HDFCLIFE": "HDFC Life", "HEROMOTOCO": "Hero MotoCorp",
    "HINDALCO": "Hindalco", "HINDUNILVR": "Hindustan Unilever",
    "ICICIBANK": "ICICI Bank", "INDUSINDBK": "IndusInd Bank",
    "INFY": "Infosys", "ITC": "ITC", "JIOFIN": "Jio Financial",
    "JSWSTEEL": "JSW Steel", "KOTAKBANK": "Kotak Mahindra Bank",
    "LT": "Larsen & Toubro", "M&M": "Mahindra & Mahindra",
    "MARUTI": "Maruti Suzuki", "NESTLEIND": "Nestle India", "NTPC": "NTPC",
    "ONGC": "ONGC", "POWERGRID": "Power Grid", "RELIANCE": "Reliance Industries",
    "SBILIFE": "SBI Life", "SBIN": "State Bank of India",
    "SHRIRAMFIN": "Shriram Finance", "SUNPHARMA": "Sun Pharma",
    "TATACONSUM": "Tata Consumer", "TMPV": "Tata Motors",
    "TATASTEEL": "Tata Steel", "TCS": "TCS", "TECHM": "Tech Mahindra",
    "TITAN": "Titan", "TRENT": "Trent", "ULTRACEMCO": "UltraTech Cement",
    "WIPRO": "Wipro",
}


def load_universe(path):
    if not path:
        return NIFTY50
    with open(path) as f:
        syms = [line.strip().upper() for line in f if line.strip()]
    return {s: s for s in syms}  # search by symbol if no name given


def get_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update({
        "bullish": 2.5, "bearish": -2.5, "upgrade": 2.0, "downgrade": -2.0,
        "surge": 2.0, "surges": 2.0, "soars": 2.5, "rally": 2.0, "plunge": -2.5,
        "plunges": -2.5, "slump": -2.0, "tanks": -2.5, "crash": -3.0,
        "beat": 1.5, "beats": 1.5, "miss": -1.5, "misses": -1.5,
        "loss": -1.5, "fraud": -3.0, "penalty": -2.0, "buyback": 1.5,
    })
    return sia


# ---------------------------------------------------------------------------
# News buzz
# ---------------------------------------------------------------------------
def news_for(name, days):
    q = urllib.parse.quote(f'"{name}" stock when:{days}d')
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    titles = []
    for e in feed.entries:
        t = e.get("title", "")
        if " - " in t:
            t = t.rsplit(" - ", 1)[0]
        titles.append(t.strip())
    return list(dict.fromkeys(titles))  # de-duplicate, keep order


def scan_news(universe, days, sia):
    def work(item):
        sym, name = item
        try:
            titles = news_for(name, days)
        except Exception:
            titles = []
        scores = [sia.polarity_scores(t)["compound"] for t in titles]
        avg = sum(scores) / len(scores) if scores else 0.0
        top = max(zip(scores, titles), key=lambda x: abs(x[0]))[1] if titles else ""
        return {"symbol": sym, "news_count": len(titles),
                "sentiment": round(avg, 3), "key_headline": top}

    with ThreadPoolExecutor(max_workers=8) as pool:
        return pd.DataFrame(list(pool.map(work, universe.items())))


# ---------------------------------------------------------------------------
# Trading buzz (volume spike)
# ---------------------------------------------------------------------------
def scan_volume(symbols):
    tickers = [s + ".NS" for s in symbols]
    data = yf.download(tickers, period="2mo", interval="1d",
                       group_by="ticker", progress=False, threads=True)
    rows = []
    for s, t in zip(symbols, tickers):
        try:
            df = data[t].dropna()
            vol, close = df["Volume"], df["Close"]
            avg20 = vol.iloc[-21:-1].mean()
            rows.append({
                "symbol": s,
                "price": round(close.iloc[-1], 2),
                "chg_1d_%": round((close.iloc[-1] / close.iloc[-2] - 1) * 100, 2),
                "vol_spike": round(vol.iloc[-1] / avg20, 2) if avg20 else None,
            })
        except Exception:
            rows.append({"symbol": s, "price": None, "chg_1d_%": None, "vol_spike": None})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------
def buzz_rank(news_df, vol_df):
    df = news_df.merge(vol_df, on="symbol", how="left")
    # Percentile ranks so news count and volume spike are on the same 0-1 scale
    df["news_rank"] = df["news_count"].rank(pct=True)
    df["vol_rank"] = df["vol_spike"].rank(pct=True).fillna(0)
    df["buzz_score"] = (0.6 * df["news_rank"] + 0.4 * df["vol_rank"]).round(3)
    df["mood"] = pd.cut(df["sentiment"], [-1.01, -0.1, 0.1, 1.01],
                        labels=["negative", "neutral", "positive"])
    return df.sort_values("buzz_score", ascending=False).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description="Find the most talked-about NSE stocks")
    p.add_argument("--days", type=int, default=2, help="News lookback (default 2)")
    p.add_argument("--top", type=int, default=10, help="How many to show")
    p.add_argument("--file", help="Text file of NSE symbols, one per line")
    p.add_argument("--csv", help="Save full results to this CSV file")
    args = p.parse_args()

    universe = load_universe(args.file)
    print(f"Scanning {len(universe)} stocks (news from last {args.days} day(s))...\n")

    sia = get_vader()
    news_df = scan_news(universe, args.days, sia)
    vol_df = scan_volume(list(universe.keys()))
    result = buzz_rank(news_df, vol_df)

    cols = ["symbol", "buzz_score", "news_count", "vol_spike",
            "price", "chg_1d_%", "sentiment", "mood"]
    print(f"Top {args.top} most talked-about stocks "
          f"({datetime.now(timezone.utc).astimezone().strftime('%d %b %Y %H:%M')})")
    print("=" * 80)
    print(result[cols].head(args.top).to_string(index=False))

    print("\nWhat people are saying:")
    print("-" * 80)
    for _, r in result.head(args.top).iterrows():
        if r["key_headline"]:
            print(f" {r['symbol']:<11} [{r['sentiment']:+.2f}] {r['key_headline'][:62]}")

    if args.csv:
        result.to_csv(args.csv, index=False)
        print(f"\nSaved full results to {args.csv}")

    print("\nvol_spike = today's volume / 20-day average (2.0 = twice normal).")
    print("Reminder: high buzz means attention, not direction. Not financial advice.")


if __name__ == "__main__":
    main()
