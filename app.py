"""
Streamlit app: Detailed News Analysis for NSE stocks
----------------------------------------------------
Put this file in the same folder as:
    detailed_news_analysis.py
    most_talked_stocks.py

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import altair as alt
import pandas as pd
import streamlit as st
import yfinance as yf

import detailed_news_analysis_console as dna
from most_talked_stocks import NIFTY50

st.set_page_config(page_title="NSE News Analysis", page_icon="📰", layout="wide")

GREEN, RED, GREY = "#15803d", "#b91c1c", "#6b7280"
dna.log = lambda msg: None  # silence console progress lines inside the app


# ---------------------------------------------------------------------------
# Cached data loaders (news cached 15 min, prices 15 min)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_scorer(model):
    return dna.get_scorer(model)


@st.cache_data(ttl=900, show_spinner=False)
def load_name(symbol):
    return dna.company_name(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def load_news(symbol, name, days, fulltext):
    raw, _ = dna.collect(symbol, name, days, fulltext)
    return raw


@st.cache_data(ttl=900, show_spinner=False)
def load_price_context(symbol):
    return dna.price_context(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def load_history(symbol, period="6mo"):
    try:
        h = yf.Ticker(symbol).history(period=period)
        return h[["Close", "Volume"]].reset_index()
    except Exception:
        return pd.DataFrame()


def run_analysis(symbol, name, days, model, fulltext, half_life, threshold):
    scorer = load_scorer(model)
    raw = load_news(symbol, name, days, fulltext)
    if raw.empty:
        return None
    patterns = dna.name_patterns(name, symbol)
    stories = dna.cluster_stories(raw.copy())
    df = dna.analyze_stories(stories, scorer, patterns, half_life)
    agg = dna.aggregate(df, threshold)
    ctx = load_price_context(symbol)
    notes = dna.interpret(agg, ctx)
    points = dna.key_points(df, agg, ctx)
    return {"df": df, "agg": agg, "ctx": ctx, "notes": notes, "points": points,
            "scorer": scorer.name, "n_raw": len(raw)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tone_color(x):
    return GREEN if x > 0.05 else RED if x < -0.05 else GREY


def ist(ts):
    return ts.tz_convert(dna.IST).tz_localize(None) if ts is not None and pd.notna(ts) else None


def story_card(r, positive):
    with st.container(border=True):
        color = tone_color(r["sentiment"])
        title = f"[{r['title']}]({r['link']})" if r["link"] else r["title"]
        st.markdown(f"<span style='color:{color};font-weight:700'>{r['sentiment']:+.2f}</span> "
                    f"&nbsp;{title}", unsafe_allow_html=True)
        more = f" (+{r['coverage'] - 1} more sources)" if r["coverage"] > 1 else ""
        st.caption(f"{r['event']}  |  {r['source'] or 'Unknown source'}{more}  |  "
                   f"{dna.fmt_time(r['published'])} IST  |  impact {r['impact']:+.2f}")
        line = r["best_positive_line"] if positive else r["best_negative_line"]
        if line and line.strip(". ").lower() != r["title"].lower():
            st.markdown(f"> {line[:240]}")
        if r["figures"]:
            st.markdown(f"**Figures:** {r['figures']}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Stock")
    options = ["Type a symbol…"] + [f"{s} ({n})" for s, n in sorted(NIFTY50.items())]
    choice = st.selectbox("Nifty 50 stock", options, index=options.index("RELIANCE (Reliance Industries)"))
    if choice == "Type a symbol…":
        ticker = st.text_input("NSE symbol", placeholder="e.g. ZYDUSLIFE").strip().upper()
        default_name = ""
    else:
        ticker = choice.split(" ")[0]
        default_name = NIFTY50[ticker]
    name_override = st.text_input("Company name for news search", value=default_name,
                                  help="Change this if the search finds the wrong company.")

    st.header("Settings")
    days = st.slider("News from the last N days", 1, 30, 7)
    model = st.radio("Sentiment model", ["vader", "finbert"],
                     format_func=lambda m: "VADER (fast)" if m == "vader" else "FinBERT (accurate, slower)")
    fulltext = st.checkbox("Read full articles (slower)", value=False)
    with st.expander("Advanced"):
        half_life = st.slider("Recency half-life (hours)", 6, 120, 36,
                              help="A story loses half its weight after this many hours.")
        threshold = st.slider("Score needed for UP / DOWN", 0.05, 0.50, 0.15, 0.01)

    analyze = st.button("Analyze news", type="primary", width="stretch", disabled=not ticker)
    st.caption("Results are cached for 15 minutes.")

if analyze:
    st.session_state["params"] = (ticker, name_override, days, model, fulltext, half_life, threshold)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("NSE stock news analysis")

if "params" not in st.session_state:
    st.write("Pick a stock in the sidebar and select **Analyze news** to see what the news "
             "says about it: the overall tone, what's driving it, and the stories that matter most.")
    st.stop()

ticker, name_override, days, model, fulltext, half_life, threshold = st.session_state["params"]
symbol = dna.to_nse_symbol(ticker)
name = name_override.strip() or load_name(symbol)

try:
    with st.spinner(f"Reading news about {name}…"):
        res = run_analysis(symbol, name, days, model, fulltext, half_life, threshold)
except Exception as e:
    st.error(f"Analysis failed: {e}. Check your internet connection and the symbol, then try again.")
    st.stop()

if res is None:
    st.warning(f"No news found for {name} ({symbol}) in the last {days} days. "
               "Try a longer period, or change the company name in the sidebar.")
    st.stop()

df, agg, ctx = res["df"], res["agg"], res["ctx"]

st.subheader(f"{name} ({symbol})")
st.caption(f"{len(df)} unique stories from {res['n_raw']} articles  |  last {days} days  |  "
           f"model: {res['scorer']}")

# --- headline metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Verdict", agg["signal"].split(" ")[0], agg["signal"].split(" ", 1)[1].strip("()")
          if " " in agg["signal"] else None, delta_color="off")
c2.metric("Sentiment", f"{agg['score']:+.2f}", help="-1 = very negative, +1 = very positive")
c3.metric("Confidence", agg["conf_label"], f"{agg['confidence']:.2f}", delta_color="off",
          help="Higher when there are more stories and they agree with each other.")
if ctx:
    c4.metric("Price", f"₹{ctx['price']:,.2f}", f"{ctx['chg_1d']:+.2f}% today")
    rel = f"{ctx['rel_5d']:+.1f}% vs Nifty" if "rel_5d" in ctx else None
    c5.metric("5-day change", f"{ctx['chg_5d']:+.2f}%", rel)

# --- key points + interpretation
left, right = st.columns([3, 2])
with left:
    st.markdown("#### Key points")
    st.markdown("\n".join(f"- {p}" for p in res["points"]))
with right:
    st.markdown("#### What it means")
    if not res["notes"]:
        st.write("Nothing unusual stands out.")
    for n in res["notes"]:
        (st.warning if n.startswith(("Divergence", "Low confidence")) else st.info)(n)

# --- tabs
t_news, t_events, t_trend, t_price, t_all = st.tabs(
    ["Top news", "By event type", "Daily trend", "Price vs news", "All stories"])

with t_news:
    cp, cn = st.columns(2)
    with cp:
        st.markdown(f"#### <span style='color:{GREEN}'>Positive</span>", unsafe_allow_html=True)
        if agg["top_pos"].empty:
            st.write("No positive stories.")
        for _, r in agg["top_pos"].iterrows():
            story_card(r, True)
    with cn:
        st.markdown(f"#### <span style='color:{RED}'>Negative</span>", unsafe_allow_html=True)
        if agg["top_neg"].empty:
            st.write("No negative stories.")
        for _, r in agg["top_neg"].iterrows():
            story_card(r, False)

with t_events:
    ev = agg["by_event"].reset_index()
    ev["tone"] = ev["total_impact"].apply(lambda x: "Positive" if x > 0 else "Negative")
    chart = (alt.Chart(ev).mark_bar().encode(
        x=alt.X("total_impact:Q", title="Total impact (sentiment × weight)"),
        y=alt.Y("event:N", sort="-x", title=None),
        color=alt.Color("tone:N", scale=alt.Scale(domain=["Positive", "Negative"], range=[GREEN, RED]),
                        legend=None),
        tooltip=["event", "stories", alt.Tooltip("avg_sentiment:Q", format="+.2f"),
                 alt.Tooltip("share_of_impact_%:Q", format=".0f", title="share %")])
             .properties(height=max(160, 38 * len(ev))))
    st.altair_chart(chart, width="stretch")
    st.caption("Events that matter more for prices (results, regulatory action, analyst calls) carry "
               "more weight than reports that a share simply rose or fell.")
    st.dataframe(ev.drop(columns="tone").rename(columns={
        "event": "Event", "stories": "Stories", "avg_sentiment": "Avg sentiment",
        "total_impact": "Impact", "share_of_impact_%": "Share %"}).round(2),
        hide_index=True, width="stretch")

with t_trend:
    daily = agg["daily"].reset_index()
    if daily.empty:
        st.write("No dated stories to plot.")
    else:
        daily["date"] = pd.to_datetime(daily["date"])
        daily["tone"] = daily["avg_sentiment"].apply(
            lambda x: "Positive" if x > 0.05 else "Negative" if x < -0.05 else "Neutral")
        bars = (alt.Chart(daily).mark_bar(size=22).encode(
            x=alt.X("yearmonthdate(date):T", title=None),
            y=alt.Y("avg_sentiment:Q", title="Average sentiment", scale=alt.Scale(domain=[-1, 1])),
            color=alt.Color("tone:N", scale=alt.Scale(domain=["Positive", "Neutral", "Negative"],
                                                      range=[GREEN, GREY, RED]), legend=None),
            tooltip=[alt.Tooltip("yearmonthdate(date):T", title="Date"), "stories", "positive", "negative",
                     alt.Tooltip("avg_sentiment:Q", format="+.2f")]))
        st.altair_chart(bars.properties(height=300), width="stretch")
        st.caption("Bar height is the day's average sentiment. Hover a bar to see the number of stories.")

with t_price:
    hist = load_history(symbol)
    if hist.empty:
        st.write("Price data isn't available right now.")
    else:
        hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None).dt.normalize()
        line = alt.Chart(hist).mark_line(color="#334155").encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Close:Q", title="Close (₹)", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Close:Q", format=",.2f")])
        layers = [line]
        daily = agg["daily"].reset_index()
        if not daily.empty:
            daily["Date"] = pd.to_datetime(daily["date"])
            marks = daily.merge(hist, on="Date", how="inner")
            if not marks.empty:
                marks["tone"] = marks["avg_sentiment"].apply(
                    lambda x: "Positive" if x > 0.05 else "Negative" if x < -0.05 else "Neutral")
                layers.append(alt.Chart(marks).mark_circle(size=140, opacity=0.9).encode(
                    x="Date:T", y="Close:Q",
                    color=alt.Color("tone:N", title="News tone",
                                    scale=alt.Scale(domain=["Positive", "Neutral", "Negative"],
                                                    range=[GREEN, GREY, RED])),
                    tooltip=[alt.Tooltip("Date:T"), "stories",
                             alt.Tooltip("avg_sentiment:Q", format="+.2f", title="sentiment")]))
        st.altair_chart(alt.layer(*layers).properties(height=340), width="stretch")
        st.caption("Dots mark trading days with news, coloured by that day's tone. "
                   "News on weekends and holidays has no dot.")

with t_all:
    table = df.sort_values("impact", key=abs, ascending=False).copy()
    table["published"] = table["published"].apply(ist)
    table["sources"] = table["sources"].apply(lambda s: ", ".join(s))
    show = table[["sentiment", "title", "event", "source", "published", "coverage",
                  "weight", "impact", "figures", "link"]]
    st.dataframe(
        show, hide_index=True, width="stretch", height=480,
        column_config={
            "sentiment": st.column_config.NumberColumn("Sentiment", format="%+.2f"),
            "title": st.column_config.TextColumn("Headline", width="large"),
            "event": "Event", "source": "Source",
            "published": st.column_config.DatetimeColumn("First seen (IST)", format="DD MMM, HH:mm"),
            "coverage": st.column_config.NumberColumn("Coverage", help="Outlets carrying this story"),
            "weight": st.column_config.NumberColumn("Weight", format="%.2f"),
            "impact": st.column_config.NumberColumn("Impact", format="%+.2f"),
            "figures": "Key figures",
            "link": st.column_config.LinkColumn("Link", display_text="Open"),
        })
    st.download_button("Download stories as CSV",
                       show.to_csv(index=False).encode("utf-8"),
                       file_name=f"{ticker}_news.csv", mime="text/csv")

st.divider()
st.caption("Sentiment reflects the tone of news coverage, not a prediction of future prices. "
           "Not financial advice.")
