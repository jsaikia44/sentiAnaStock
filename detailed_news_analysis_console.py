"""
Detailed News Analysis for NSE Stocks
-------------------------------------
A deeper version of stock_sentiment.py. For one stock it:

  1. Collects news from Google News India, major Indian publisher RSS feeds
     (Economic Times, Moneycontrol, Mint, Business Standard) and Yahoo Finance
  2. Groups duplicate stories (same news rewritten by many outlets)
  3. Classifies each story by EVENT TYPE (earnings, analyst call, order win,
     regulatory, management, M&A, dividend, promoter/FII activity, ...)
  4. Scores the headline AND the article body sentence by sentence
  5. Weights every story by event importance, source quality, relevance,
     recency and how widely it was covered
  6. Extracts key figures (target prices, % moves, Rs crore amounts)
  7. Shows a day-by-day sentiment trend, the main positive/negative drivers,
     and compares the news mood with the actual price move
  8. Prints a short KEY POINTS summary in the console (--full for more detail)

Install:
    pip install yfinance pandas nltk feedparser
    pip install trafilatura              # optional: full article text (--fulltext)
    pip install transformers torch       # optional: FinBERT model (--model finbert)

Usage:
    python detailed_news_analysis.py RELIANCE
    python detailed_news_analysis.py HDFCBANK --days 14 --fulltext
    python detailed_news_analysis.py TMPV --name "Tata Motors" --model finbert
    python detailed_news_analysis.py INFY --full      # add event table + all stories

DISCLAIMER: News sentiment is a noisy signal and does not reliably predict
prices. For research/learning only. Not financial advice.
"""

import argparse
import html
import math
import re
import urllib.parse
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher

import feedparser
import pandas as pd
import yfinance as yf

IST = "Asia/Kolkata"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Publisher RSS feeds (URLs occasionally change; failures are skipped silently)
PUBLISHER_FEEDS = {
    "Economic Times": [
        "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    ],
    "Moneycontrol": [
        "https://www.moneycontrol.com/rss/business.xml",
        "https://www.moneycontrol.com/rss/marketreports.xml",
        "https://www.moneycontrol.com/rss/buzzingstocks.xml",
    ],
    "Mint": ["https://www.livemint.com/rss/markets", "https://www.livemint.com/rss/companies"],
    "Business Standard": [
        "https://www.business-standard.com/rss/markets-106.rss",
        "https://www.business-standard.com/rss/companies-101.rss",
    ],
}

# Source quality weights (anything not listed gets DEFAULT_SOURCE_WEIGHT)
TRUSTED_SOURCES = {
    "reuters": 1.0, "bloomberg": 1.0, "economic times": 1.0, "the economic times": 1.0,
    "moneycontrol": 1.0, "mint": 1.0, "livemint": 1.0, "business standard": 1.0,
    "cnbc": 0.95, "cnbctv18": 0.95, "financial express": 0.9, "the hindu businessline": 0.9,
    "businessline": 0.9, "ndtv profit": 0.9, "zee business": 0.8, "business today": 0.85,
    "the hindu": 0.85, "times of india": 0.8, "yahoo": 0.8,
}
DEFAULT_SOURCE_WEIGHT = 0.65

# Event types: (name, importance weight, regex patterns). First match = primary event.
EVENT_RULES = [
    ("Regulatory / Legal", 1.6, [r"\bsebi\b", r"\bprobe\b", r"\braid", r"\bpenalt", r"\bfine[ds]?\b",
                                 r"\bcourt\b", r"\bnclt\b", r"\blawsuit", r"\bfraud", r"\bshow[- ]cause",
                                 r"\bed\b", r"\bcbi\b", r"\bincome tax\b", r"\bgst notice", r"\bban\b"]),
    ("Earnings / Results", 1.5, [r"\bq[1-4]\b", r"\bresults?\b", r"\bprofit", r"\brevenue", r"\bearnings",
                                 r"\bebitda", r"\bmargin", r"\bnet income", r"\bpat\b", r"\bguidance"]),
    ("M&A / Stake deal", 1.4, [r"\bacqui", r"\bmerger", r"\bmerge\b", r"\bdemerg", r"\btakeover",
                               r"\bstake\b", r"\bjoint venture", r"\bjv\b", r"\bbuyout"]),
    ("Analyst rating / Target", 1.3, [r"\bupgrade", r"\bdowngrade", r"\btarget\b", r"\brating\b",
                                      r"\bbuy call", r"\bsell call", r"\boverweight", r"\bunderweight",
                                      r"\boutperform", r"\bunderperform", r"\bbrokerage", r"\bmorgan stanley",
                                      r"\bgoldman", r"\bjefferies", r"\bclsa\b", r"\bnomura", r"\bmotilal"]),
    ("Order / Contract win", 1.3, [r"\border win", r"\bbags?\b", r"\bwins?\b.*\b(order|contract|deal)",
                                   r"\bcontract\b", r"\border worth", r"\bletter of award", r"\bloa\b"]),
    ("Management / Governance", 1.2, [r"\bceo\b", r"\bcfo\b", r"\bmd\b", r"\bchairman", r"\bresign",
                                      r"\bappoint", r"\bsteps down", r"\bboard\b", r"\bauditor"]),
    ("Promoter / FII / Block deal", 1.2, [r"\bpromoter", r"\bpledge", r"\bblock deal", r"\bbulk deal",
                                          r"\bfii\b", r"\bfpi\b", r"\bdii\b", r"\bmutual funds? (buy|sell|hike|cut)",
                                          r"\bstake sale", r"\bofs\b", r"\bqip\b"]),
    ("Dividend / Buyback / Split", 1.1, [r"\bdividend", r"\bbuyback", r"\bbonus", r"\bstock split",
                                         r"\brecord date", r"\brights issue"]),
    ("Product / Expansion", 1.0, [r"\blaunch", r"\bexpan", r"\bnew plant", r"\bcapex", r"\bcapacity",
                                  r"\bpartnership", r"\bties up", r"\bmou\b", r"\bsales\b.*\b(rise|fall|jump|decline)"]),
    ("Price-move report", 0.6, [r"\bshares? (rise|rises|fall|falls|jump|jumps|surge|surges|slump|slumps|tank|tanks|gain|gains|drop|drops|climb|climbs|rally|rallies|plunge|plunges)",
                                r"\bstock (rises|falls|jumps|surges|slumps|tanks|gains|drops)",
                                r"\b52-week", r"\bupper circuit", r"\blower circuit", r"\btop (gainer|loser)"]),
    ("Macro / Sector", 0.8, [r"\brbi\b", r"\brepo\b", r"\binflation", r"\bbudget\b", r"\btariff",
                             r"\bcrude", r"\brupee", r"\bsector\b", r"\bnifty\b", r"\bsensex\b"]),
]
DEFAULT_EVENT = ("General news", 0.9)

FINANCE_LEXICON = {
    "bullish": 2.5, "bearish": -2.5, "upgrade": 2.0, "upgrades": 2.0, "upgraded": 2.0,
    "downgrade": -2.0, "downgrades": -2.0, "downgraded": -2.0, "outperform": 2.0,
    "underperform": -2.0, "overweight": 1.5, "underweight": -1.5, "beat": 1.5, "beats": 1.5,
    "miss": -1.5, "misses": -1.5, "missed": -1.5, "surge": 2.0, "surges": 2.0, "soars": 2.5,
    "jumps": 1.5, "rally": 2.0, "rallies": 2.0, "climbs": 1.2, "gains": 1.2, "plunge": -2.5,
    "plunges": -2.5, "slump": -2.0, "slumps": -2.0, "tanks": -2.5, "crash": -3.0, "slides": -1.5,
    "drops": -1.2, "falls": -1.2, "declines": -1.2, "profit": 1.0, "loss": -1.5, "losses": -1.5,
    "dividend": 1.0, "buyback": 1.5, "bonus": 1.0, "penalty": -2.0, "probe": -2.0, "fraud": -3.0,
    "default": -2.5, "pledge": -1.0, "bags": 1.5, "headwinds": -1.5, "tailwinds": 1.5,
    "record": 1.0, "robust": 2.0, "weak": -1.8, "subdued": -1.2, "muted": -1.0, "resigns": -1.5,
    "notice": -1.0, "show-cause": -1.5, "alleged": -1.5, "allegedly": -1.5, "lapses": -1.5,
    "lapse": -1.5, "violation": -2.0, "violations": -2.0, "irregularities": -2.0,
    "summons": -1.5, "raided": -2.0, "raids": -2.0, "fined": -2.0, "writedown": -2.0,
    "impairment": -1.5, "delay": -1.0, "delayed": -1.0, "stake sale": -0.5,
}


# ---------------------------------------------------------------------------
# Sentiment models
# ---------------------------------------------------------------------------
class VaderScorer:
    name = "VADER (+ finance words)"

    def __init__(self):
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        self.sia = SentimentIntensityAnalyzer()
        self.sia.lexicon.update(FINANCE_LEXICON)

    def score(self, texts):
        return [self.sia.polarity_scores(t)["compound"] for t in texts]


class FinBertScorer:
    name = "FinBERT"

    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)

    def score(self, texts):
        texts = list(texts)
        if not texts:
            return []
        out = []
        for res in self.pipe(texts, truncation=True, batch_size=16):
            p = {r["label"].lower(): r["score"] for r in res}
            out.append(p.get("positive", 0) - p.get("negative", 0))
        return out


def get_scorer(name):
    if name == "finbert":
        try:
            return FinBertScorer()
        except Exception as e:
            print(f"[!] FinBERT unavailable ({e}); using VADER.")
    return VaderScorer()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg):
    """Short progress line that is overwritten, so only the report stays on screen."""
    print(f"\r{msg[:78]:<78}", end="", flush=True)


def to_nse_symbol(t):
    t = t.upper().strip()
    return t if t.startswith("^") or t.endswith((".NS", ".BO")) else t + ".NS"


def clean_html(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def company_name(symbol, override=None):
    if override:
        return override
    try:
        info = yf.Ticker(symbol).info
        name = info.get("shortName") or info.get("longName") or ""
        name = re.sub(r"\b(Limited|Ltd\.?|LIMITED|LTD\.?)$", "", name).strip(" .,")
        if name:
            return name
    except Exception:
        pass
    return symbol.split(".")[0]


def name_patterns(name, symbol):
    """Regexes used to decide whether a story is actually about this company."""
    base = symbol.split(".")[0]
    pats = {re.escape(name.lower())}
    words = [w for w in re.split(r"\s+", name.lower()) if len(w) > 2]
    if len(words) >= 2:
        pats.add(re.escape(" ".join(words[:2])))
    if len(base) >= 3 and base.isalpha():
        pats.add(rf"\b{re.escape(base.lower())}\b")
    return [re.compile(p) for p in pats]


def mentions(text, patterns):
    t = (text or "").lower()
    return any(p.search(t) for p in patterns)


def to_ts(struct):
    if not struct:
        return None
    return pd.Timestamp(datetime(*struct[:6], tzinfo=timezone.utc))


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text or "")
    return [p.strip() for p in parts if 25 <= len(p.strip()) <= 400]


# ---------------------------------------------------------------------------
# 1. Collect news
# ---------------------------------------------------------------------------
def from_google(name, days):
    q = urllib.parse.quote(f'"{name}" when:{days}d')
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    rows = []
    for e in feedparser.parse(url).entries:
        title, source = e.get("title", ""), ""
        if " - " in title:
            title, source = title.rsplit(" - ", 1)
        src = e.get("source", {})
        source = src.get("title", source) if isinstance(src, dict) else source
        rows.append({"title": title.strip(), "summary": "", "source": source.strip(),
                     "published": to_ts(e.get("published_parsed")), "link": e.get("link", ""),
                     "origin": "Google News"})
    return rows


def from_publishers(patterns):
    rows = []
    for source, urls in PUBLISHER_FEEDS.items():
        for url in urls:
            try:
                feed = feedparser.parse(url)
            except Exception:
                continue
            for e in feed.entries:
                title = clean_html(e.get("title", ""))
                summary = clean_html(e.get("summary", ""))
                if mentions(title, patterns) or mentions(summary, patterns):
                    rows.append({"title": title, "summary": summary, "source": source,
                                 "published": to_ts(e.get("published_parsed")),
                                 "link": e.get("link", ""), "origin": "Publisher RSS"})
    return rows


def from_yahoo(symbol):
    try:
        raw = yf.Ticker(symbol).news or []
    except Exception:
        return []
    rows = []
    for item in raw:
        c = item.get("content", item)
        pub = None
        if c.get("pubDate"):
            pub = pd.to_datetime(c["pubDate"], utc=True)
        elif item.get("providerPublishTime"):
            pub = pd.to_datetime(item["providerPublishTime"], unit="s", utc=True)
        link = (c.get("canonicalUrl") or {}).get("url", "") if isinstance(c.get("canonicalUrl"), dict) \
            else item.get("link", "")
        provider = c.get("provider", {})
        rows.append({"title": c.get("title", ""), "summary": c.get("summary", "") or "",
                     "source": provider.get("displayName", "Yahoo") if isinstance(provider, dict) else "Yahoo",
                     "published": pub, "link": link, "origin": "Yahoo Finance"})
    return [r for r in rows if r["title"]]


def add_fulltext(df, limit=20):
    """Download article bodies for direct (non-Google) links using trafilatura."""
    try:
        import trafilatura
    except ImportError:
        print("[!] trafilatura not installed; skipping full text (pip install trafilatura)")
        return df
    df["fulltext"] = ""
    done = 0
    for i, row in df.iterrows():
        if done >= limit or not row["link"] or "news.google.com" in row["link"]:
            continue
        try:
            page = trafilatura.fetch_url(row["link"])
            text = trafilatura.extract(page) if page else None
            if text:
                df.at[i, "fulltext"] = text[:8000]
                done += 1
        except Exception:
            pass
    log(f"    full text fetched for {done} article(s)")
    return df


def collect(symbol, name, days, fulltext):
    patterns = name_patterns(name, symbol)
    log(f"[1/6] Collecting news for {name} ({symbol}) ...")
    rows = from_google(name, days) + from_publishers(patterns) + from_yahoo(symbol)
    df = pd.DataFrame(rows)
    if df.empty:
        return df, patterns
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    df = df[df["published"].isna() | (df["published"] >= cutoff)].reset_index(drop=True)
    log(f"    {len(df)} raw articles: " +
          ", ".join(f"{k} {v}" for k, v in df["origin"].value_counts().items()))
    if fulltext:
        df = add_fulltext(df)
    if "fulltext" not in df:
        df["fulltext"] = ""
    return df, patterns


# ---------------------------------------------------------------------------
# 2. Group duplicate stories
# ---------------------------------------------------------------------------
def normalize_title(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower())


def cluster_stories(df, threshold=0.62):
    log("[2/6] Grouping duplicate stories ...")
    df = df.sort_values("published", na_position="last").reset_index(drop=True)
    reps, cluster_ids = [], []
    for t in df["title"].map(normalize_title):
        cid = None
        for k, rep in enumerate(reps):
            if SequenceMatcher(None, t, rep).ratio() >= threshold:
                cid = k
                break
        if cid is None:
            reps.append(t)
            cid = len(reps) - 1
        cluster_ids.append(cid)
    df["cluster"] = cluster_ids

    stories = []
    for cid, g in df.groupby("cluster"):
        g = g.assign(_len=g["summary"].str.len() + g["fulltext"].str.len())
        best = g.sort_values("_len", ascending=False).iloc[0]
        sources = sorted({s for s in g["source"] if s})
        stories.append({
            "title": best["title"], "summary": best["summary"], "fulltext": best["fulltext"],
            "link": best["link"], "source": best["source"] or (sources[0] if sources else ""),
            "sources": sources, "coverage": len(g),
            "published": g["published"].min(),  # first time the story appeared
        })
    out = pd.DataFrame(stories)
    log(f"    {len(df)} articles -> {len(out)} unique stories")
    return out


# ---------------------------------------------------------------------------
# 3-5. Analyze each story
# ---------------------------------------------------------------------------
def classify_event(text):
    t = text.lower()
    matched = [(name, w) for name, w, pats in EVENT_RULES if any(re.search(p, t) for p in pats)]
    if not matched:
        return DEFAULT_EVENT[0], DEFAULT_EVENT[1], []
    primary = matched[0]
    return primary[0], primary[1], [m[0] for m in matched[1:]]


def source_weight(source):
    s = (source or "").lower()
    for key, w in TRUSTED_SOURCES.items():
        if key in s:
            return w
    return DEFAULT_SOURCE_WEIGHT


def recency_weight(ts, half_life_h):
    if ts is None or pd.isna(ts):
        return 0.5
    age = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 3600
    return 0.5 ** (max(age, 0) / half_life_h)


FIGURE_PATTERNS = [
    ("Target price", re.compile(r"target(?: price)?(?: of| to| at| raised to| cut to)?\s*(?:rs\.?|₹|inr)\s?([\d,]+(?:\.\d+)?)", re.I)),
    ("Amount", re.compile(r"((?:rs\.?|₹|inr)\s?[\d,]+(?:\.\d+)?\s?(?:crore|cr|lakh crore|billion|million|bn|mn))", re.I)),
    ("% change", re.compile(r"((?:up|down|rose|fell|jumped|declined|grew|surged|dropped|gained|slipped)\s(?:by\s)?\d+(?:\.\d+)?\s?(?:%|per ?cent))", re.I)),
]


def extract_figures(text):
    found = []
    for label, pat in FIGURE_PATTERNS:
        for m in pat.findall(text or "")[:3]:
            val = m if isinstance(m, str) else m[0]
            val = ("₹" + val) if label == "Target price" else val
            found.append(f"{label}: {val.strip()}")
    return list(dict.fromkeys(found))[:5]


def analyze_stories(stories, scorer, patterns, half_life_h):
    log("[3/6] Classifying events and scoring sentiment ...")
    stories["headline_score"] = scorer.score(stories["title"].tolist())

    # Sentence-level body scoring (prefer sentences that mention the company)
    all_sents, owners = [], []
    for i, r in stories.iterrows():
        body = r["fulltext"] or r["summary"]
        sents = split_sentences(body)
        about = [s for s in sents if mentions(s, patterns)]
        chosen = (about or sents)[:12]
        all_sents += chosen
        owners += [i] * len(chosen)
    sent_scores = scorer.score(all_sents) if all_sents else []

    body_score, best_pos, best_neg, n_sents = {}, {}, {}, {}
    for i, s, sc in zip(owners, all_sents, sent_scores):
        body_score.setdefault(i, []).append(sc)
        if sc > best_pos.get(i, (0.05, ""))[0]:
            best_pos[i] = (sc, s)
        if sc < best_neg.get(i, (-0.05, ""))[0]:
            best_neg[i] = (sc, s)

    rows = []
    for i, r in stories.iterrows():
        text_all = f"{r['title']}. {r['summary']} {r['fulltext'][:1500]}"
        event, ev_w, secondary = classify_event(f"{r['title']}. {r['summary']}")
        b = body_score.get(i)
        body = sum(b) / len(b) if b else None
        sentiment = r["headline_score"] if body is None else 0.6 * r["headline_score"] + 0.4 * body
        relevance = 1.0 if mentions(r["title"], patterns) else 0.5
        src_w = source_weight(r["source"])
        rec_w = recency_weight(r["published"], half_life_h)
        cov_w = min(1 + 0.2 * math.log2(r["coverage"]), 1.8)
        weight = ev_w * relevance * src_w * rec_w * cov_w
        rows.append({
            "event": event, "other_tags": ", ".join(secondary), "event_weight": ev_w,
            "body_score": body, "n_sentences": len(b) if b else 0, "sentiment": sentiment,
            "relevance": relevance, "source_weight": src_w, "recency_weight": rec_w,
            "coverage_weight": cov_w, "weight": weight, "impact": sentiment * weight,
            "figures": "; ".join(extract_figures(text_all)),
            "best_positive_line": best_pos.get(i, (None, ""))[1],
            "best_negative_line": best_neg.get(i, (None, ""))[1],
        })
    return pd.concat([stories, pd.DataFrame(rows, index=stories.index)], axis=1)


# ---------------------------------------------------------------------------
# 6. Aggregate
# ---------------------------------------------------------------------------
def aggregate(df, threshold):
    log("[4/6] Aggregating ...")
    w = df["weight"]
    score = (df["sentiment"] * w).sum() / w.sum()
    spread = math.sqrt(((df["sentiment"] - score) ** 2 * w).sum() / w.sum())
    agreement = max(0.0, 1 - spread)               # 1 = all stories agree
    volume_factor = min(1.0, len(df) / 15)          # more stories = more reliable
    confidence = agreement * 0.6 + volume_factor * 0.4
    conf_label = "High" if confidence > 0.7 else "Medium" if confidence > 0.45 else "Low"

    signal = ("UP (bullish)" if score > threshold else
              "DOWN (bearish)" if score < -threshold else "NEUTRAL")

    by_event = (df.groupby("event")
                .agg(stories=("title", "count"), avg_sentiment=("sentiment", "mean"),
                     total_impact=("impact", "sum"))
                .sort_values("total_impact", key=abs, ascending=False))
    by_event["share_of_impact_%"] = (by_event["total_impact"].abs() /
                                     by_event["total_impact"].abs().sum() * 100)

    d = df.dropna(subset=["published"]).copy()
    d["date"] = d["published"].dt.tz_convert(IST).dt.date
    daily = (d.groupby("date").agg(stories=("title", "count"), avg_sentiment=("sentiment", "mean"),
                                   positive=("sentiment", lambda s: (s > 0.05).sum()),
                                   negative=("sentiment", lambda s: (s < -0.05).sum()))
             .sort_index())

    source_mix = Counter(s for lst in df["sources"] for s in lst).most_common(8)

    return {"score": score, "signal": signal, "spread": spread, "confidence": confidence,
            "conf_label": conf_label, "by_event": by_event, "daily": daily,
            "source_mix": source_mix,
            "top_pos": df[df["impact"] > 0].nlargest(5, "impact"),
            "top_neg": df[df["impact"] < 0].nsmallest(5, "impact")}


def price_context(symbol):
    log("[5/6] Fetching price context ...")
    try:
        h = yf.Ticker(symbol).history(period="3mo")
        n = yf.Ticker("^NSEI").history(period="3mo")
    except Exception:
        return None
    if len(h) < 22:
        return None
    c, v = h["Close"], h["Volume"]
    pct = lambda s, k: (s.iloc[-1] / s.iloc[-1 - k] - 1) * 100
    ctx = {"price": c.iloc[-1], "chg_1d": pct(c, 1), "chg_5d": pct(c, 5), "chg_1m": pct(c, 21),
           "vol_spike": v.iloc[-1] / v.iloc[-21:-1].mean() if v.iloc[-21:-1].mean() else None,
           "high_3m": c.max(), "low_3m": c.min()}
    if len(n) > 6:
        ctx["nifty_5d"] = pct(n["Close"], 5)
        ctx["rel_5d"] = ctx["chg_5d"] - ctx["nifty_5d"]
    return ctx


def interpret(agg, ctx):
    notes = []
    s = agg["score"]
    if agg["conf_label"] == "Low":
        notes.append("Low confidence: few stories or stories disagree strongly; treat the signal as weak.")
    if ctx:
        if s > 0.15 and ctx["chg_5d"] < -2:
            notes.append("Divergence: news is positive but the stock fell over 5 days. "
                         "The market may be ignoring the news or expected even better.")
        elif s < -0.15 and ctx["chg_5d"] > 2:
            notes.append("Divergence: news is negative but the stock rose over 5 days. "
                         "Bad news may already be priced in.")
        elif abs(s) > 0.15 and ((s > 0) == (ctx["chg_5d"] > 0)):
            notes.append("News mood and recent price move agree. Note that some headlines "
                         "may simply be reporting the move rather than causing it.")
        if ctx.get("vol_spike") and ctx["vol_spike"] > 2:
            notes.append(f"Volume is {ctx['vol_spike']:.1f}x normal: unusual activity today.")
    pm = agg["by_event"].loc["Price-move report"]["share_of_impact_%"] \
        if "Price-move report" in agg["by_event"].index else 0
    if pm > 40:
        notes.append("Much of the coverage just reports price moves (reactive news), "
                     "which says little about what comes next.")
    return notes


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def fmt_time(ts):
    return ts.tz_convert(IST).strftime("%d-%b %H:%M") if ts is not None and pd.notna(ts) else "unknown"


def bar(x, width=20):
    n = int(round(abs(x) * width))
    return ("+" * n if x >= 0 else "-" * n).ljust(width)


def key_points(df, agg, ctx):
    """Turn the analysis into a handful of plain-language bullet points."""
    pts = []
    ev = agg["by_event"]
    total = len(df)

    # What is driving the news
    drivers = []
    for name, r in ev.head(2).iterrows():
        if r["share_of_impact_%"] < 10:
            continue
        mood = "positive" if r["avg_sentiment"] > 0.05 else "negative" if r["avg_sentiment"] < -0.05 else "mixed"
        drivers.append(f"{name} ({mood}, {r['share_of_impact_%']:.0f}% of impact)")
    if drivers:
        pts.append("Main drivers: " + "; ".join(drivers) + ".")

    # Balance of stories
    pos, neg = (df["sentiment"] > 0.05).sum(), (df["sentiment"] < -0.05).sum()
    pts.append(f"{pos} positive, {neg} negative, {total - pos - neg} neutral stories.")

    # Biggest positive / negative story
    if not agg["top_pos"].empty:
        r = agg["top_pos"].iloc[0]
        pts.append(f"Biggest positive: {r['title'][:90]} ({r['event']}).")
    if not agg["top_neg"].empty:
        r = agg["top_neg"].iloc[0]
        pts.append(f"Biggest negative: {r['title'][:90]} ({r['event']}).")

    # Analyst targets mentioned
    targets = sorted({f for fs in df["figures"] for f in fs.split("; ") if f.startswith("Target price")})
    if targets:
        pts.append("Analyst targets mentioned: " + ", ".join(t.split(": ")[1] for t in targets[:4])
                   + (f" (current price ₹{ctx['price']:,.0f})." if ctx else "."))

    # Trend over the period
    daily = agg["daily"]
    if len(daily) >= 2:
        last = daily["avg_sentiment"].iloc[-1]
        before = daily["avg_sentiment"].iloc[:-1].mean()
        if last - before > 0.15:
            pts.append("Sentiment is improving: the latest day is more positive than earlier days.")
        elif before - last > 0.15:
            pts.append("Sentiment is worsening: the latest day is more negative than earlier days.")
        else:
            pts.append("Sentiment has been fairly steady over the period.")
    return pts


def print_report(symbol, name, df, agg, ctx, notes, scorer_name, days, full=False):
    print("\r" + " " * 80 + "\r", end="")
    L = "=" * 80
    print(L)
    print(f" {name} ({symbol}) | news from last {days} days | model: {scorer_name}")
    print(L)
    print(f" VERDICT    : {agg['signal']}   sentiment {agg['score']:+.2f}   "
          f"confidence {agg['conf_label']}")
    if ctx:
        nifty = f" (Nifty {ctx['nifty_5d']:+.1f}%)" if "nifty_5d" in ctx else ""
        vol = f" | volume {ctx['vol_spike']:.1f}x" if ctx.get("vol_spike") else ""
        print(f" PRICE      : ₹{ctx['price']:,.2f} | 1d {ctx['chg_1d']:+.1f}% | "
              f"5d {ctx['chg_5d']:+.1f}%{nifty} | 1m {ctx['chg_1m']:+.1f}%{vol}")
    print(f" BASED ON   : {len(df)} unique stories ({int(df['coverage'].sum())} articles)")

    print("\n KEY POINTS")
    for i, pt in enumerate(key_points(df, agg, ctx), 1):
        print(f"  {i}. {pt}")

    for title, part, sign in (("POSITIVE NEWS", agg["top_pos"], "+"),
                              ("NEGATIVE NEWS", agg["top_neg"], "-")):
        print(f"\n {title}")
        if part.empty:
            print("  (none)")
            continue
        for _, r in part.head(3).iterrows():
            src = f"{r['source']}" + (f" +{r['coverage'] - 1} more" if r["coverage"] > 1 else "")
            print(f"  {sign} [{r['sentiment']:+.2f}] {r['title'][:72]}")
            print(f"      {r['event']} | {src} | {fmt_time(r['published'])}")
            line = r["best_positive_line"] if sign == "+" else r["best_negative_line"]
            if line and line.strip(". ").lower() != r["title"].lower():
                print(f"      > {line[:110]}")
            if r["figures"]:
                print(f"      # {r['figures'][:110]}")

    if notes:
        print("\n WHAT IT MEANS")
        for n in notes:
            print(f"  • {n}")

    if full:
        print("\n BY EVENT TYPE")
        print(f"  {'Event':<30}{'Stories':>8}{'Avg':>8}{'Impact':>9}{'Share':>8}")
        for ev, r in agg["by_event"].iterrows():
            print(f"  {ev:<30}{int(r['stories']):>8}{r['avg_sentiment']:>+8.2f}"
                  f"{r['total_impact']:>+9.2f}{r['share_of_impact_%']:>7.0f}%")
        print("\n DAILY TREND (IST)")
        for d, r in agg["daily"].iterrows():
            print(f"  {d.strftime('%a %d-%b')}  {int(r['stories']):>3} stories  "
                  f"{r['avg_sentiment']:+.2f} [{bar(r['avg_sentiment'], 16)}]")
        print("\n ALL STORIES (by impact)")
        for _, r in df.sort_values("impact", key=abs, ascending=False).iterrows():
            print(f"  [{r['sentiment']:+.2f}] {fmt_time(r['published'])}  {r['title'][:62]}")

    print("\n Not financial advice. Sentiment reflects the tone of coverage, not future prices.")
    print(L)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Detailed news analysis for an NSE stock")
    p.add_argument("ticker", help="NSE symbol, e.g. RELIANCE")
    p.add_argument("--name", help="Company name to search (if auto-detection is wrong)")
    p.add_argument("--days", type=int, default=7, help="News lookback in days (default 7)")
    p.add_argument("--model", choices=["vader", "finbert"], default="vader")
    p.add_argument("--fulltext", action="store_true", help="Download full article text (slower)")
    p.add_argument("--full", action="store_true", help="Also show event table, daily trend, all stories")
    p.add_argument("--half-life", type=float, default=36.0, help="Recency half-life in hours")
    p.add_argument("--threshold", type=float, default=0.15, help="Score needed for UP/DOWN")
    args = p.parse_args()

    symbol = to_nse_symbol(args.ticker)
    name = company_name(symbol, args.name)
    scorer = get_scorer(args.model)

    raw, patterns = collect(symbol, name, args.days, args.fulltext)
    if raw.empty:
        print("\nNo news found. Try --days 14 or --name \"Full Company Name\".")
        return

    stories = cluster_stories(raw)
    df = analyze_stories(stories, scorer, patterns, args.half_life)
    agg = aggregate(df, args.threshold)
    ctx = price_context(symbol)
    notes = interpret(agg, ctx)
    print_report(symbol, name, df, agg, ctx, notes, scorer.name, args.days, args.full)


if __name__ == "__main__":
    main()
