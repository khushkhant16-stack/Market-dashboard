import os
import re
import time
from datetime import datetime, timezone, timedelta
from urllib.parse import quote_plus

import feedparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ============================================================
# Version 1: iPad-first, zero-cost Streamlit equity dashboard
# ------------------------------------------------------------
# This app is intentionally simple and defensive.
# It uses free data sources only. Some fields will be unavailable,
# delayed, cached, rate-limited, or incomplete. When that happens,
# the app shows N/A instead of making up data.
# ============================================================

st.set_page_config(
    page_title="Zero-Cost Equity Research Dashboard V1",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DISCLAIMER = (
    "This dashboard uses free data sources. Quotes, news, options, filings, and fundamentals may be "
    "delayed, incomplete, cached, rate-limited, or unavailable. This is a research tool, not a "
    "professional trading terminal. This is not financial advice."
)

DEFAULT_PORTFOLIO = "MU, MRVL, NVDA, AMD, QQQ, SPY"

MARKET_TICKERS = [
    "SPY", "QQQ", "DIA", "IWM", "TLT", "IEF", "SHY", "UUP", "USO", "XLE",
    "SMH", "SOXX", "NVDA", "AMD", "BTC-USD", "^VIX"
]

DEFAULT_UNIVERSE = [
    # Mega-cap / AI / software
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL", "CRM",
    # Semiconductors / AI infrastructure
    "AMD", "MU", "MRVL", "SMCI", "ARM", "TSM", "ASML", "LRCX", "KLAC", "AMAT", "INTC", "QCOM", "NXPI",
    # Cybersecurity / software
    "CRWD", "PANW", "ZS", "NET", "DDOG", "SNOW", "MDB", "PLTR", "S", "OKTA",
    # Energy / utilities / nuclear / uranium
    "XOM", "CVX", "COP", "OXY", "SLB", "NEE", "SO", "DUK", "CEG", "VST", "CCJ", "UEC", "UUUU", "SMR",
    # Defense / space / robotics
    "LMT", "RTX", "NOC", "GD", "BA", "RKLB", "ACHR", "KTOS", "IRDM", "TER",
    # Healthcare
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ISRG", "VRTX",
    # Financials
    "JPM", "BAC", "GS", "MS", "V", "MA", "BRK-B", "SCHW", "COIN", "HOOD",
    # Consumer staples / defensive
    "COST", "WMT", "PG", "KO", "PEP", "MCD", "TGT", "HD", "LOW",
    # Speculative / high beta / squeeze watch candidates
    "RIVN", "LCID", "SOFI", "UPST", "AI", "IONQ", "QBTS", "RGTI", "GME", "AMC", "MSTR", "RIOT", "MARA",
]

RSS_FEEDS = {
    "Broad market / macro": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY,QQQ,DIA,IWM&region=US&lang=en-US",
        "https://www.investing.com/rss/news_25.rss",
    ],
    "Interest rates / inflation / Fed": [
        "https://www.investing.com/rss/news_14.rss",
    ],
    "Semiconductors / AI": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA,AMD,MU,MRVL,SMH,SOXX&region=US&lang=en-US",
    ],
    "Energy / utilities / nuclear": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=XLE,USO,NEE,CEG,VST,CCJ,SMR&region=US&lang=en-US",
    ],
    "Defense / space / robotics": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=LMT,RTX,NOC,RKLB,ACHR,KTOS&region=US&lang=en-US",
    ],
    "Cybersecurity / software": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=CRWD,PANW,ZS,NET,DDOG,SNOW,PLTR&region=US&lang=en-US",
    ],
    "Earnings and guidance": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA&region=US&lang=en-US",
    ],
}

SECTOR_DEFENSIVE = {"Utilities", "Consumer Defensive", "Consumer Staples", "Healthcare", "Communication Services", "Financial Services"}

# --------------------------
# Styling
# --------------------------
st.markdown(
    """
    <style>
    .stApp { background: #0b1117; color: #eef2f7; }
    div[data-testid="stMetric"] {
        background: #111a24;
        border: 1px solid #223244;
        padding: 14px;
        border-radius: 14px;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.02);
    }
    .small-muted { color: #9aa8b7; font-size: 0.9rem; }
    .risk-box {
        background: #111a24;
        border: 1px solid #2c3f54;
        padding: 12px 14px;
        border-radius: 12px;
        margin: 8px 0px;
    }
    .warning-box {
        background: #241a11;
        border: 1px solid #6d4c1f;
        padding: 12px 14px;
        border-radius: 12px;
        margin: 8px 0px;
    }
    .good { color: #3ddc97; font-weight: 700; }
    .bad { color: #ff6b6b; font-weight: 700; }
    .neutral { color: #ffd166; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# Utility helpers
# --------------------------
def utc_now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def normalize_tickers(text, max_count=100):
    if not text:
        return []
    raw = re.split(r"[,\s]+", text.upper().strip())
    tickers = []
    for item in raw:
        item = item.strip().replace("$", "")
        if not item:
            continue
        if item not in tickers:
            tickers.append(item)
    return tickers[:max_count]


def safe_float(value):
    try:
        if value is None:
            return np.nan
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return np.nan


def fmt_num(value, decimals=2):
    value = safe_float(value)
    if np.isnan(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def fmt_pct(value, decimals=2):
    value = safe_float(value)
    if np.isnan(value):
        return "N/A"
    return f"{value:.{decimals}f}%"


def fmt_money(value):
    value = safe_float(value)
    if np.isnan(value):
        return "N/A"
    abs_value = abs(value)
    if abs_value >= 1_000_000_000_000:
        return f"${value/1_000_000_000_000:.2f}T"
    if abs_value >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    return f"${value:,.2f}"


def score_clip(x):
    return int(max(0, min(100, round(safe_float(x) if not np.isnan(safe_float(x)) else 0))))


def change_class(x):
    x = safe_float(x)
    if np.isnan(x):
        return "neutral"
    if x > 0:
        return "good"
    if x < 0:
        return "bad"
    return "neutral"


def parse_date(value):
    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple)) and len(value) > 0:
            value = value[0]
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        return pd.to_datetime(value).to_pydatetime()
    except Exception:
        return None


def days_until(value):
    dt = parse_date(value)
    if not dt:
        return np.nan
    return (dt.date() - datetime.now().date()).days

# --------------------------
# Cached data functions
# --------------------------
@st.cache_data(ttl=60, show_spinner=False)
def download_history(tickers, period="1y", interval="1d"):
    """Batch price history using yfinance. Cached to reduce free-source rate-limit pressure."""
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = [t for t in tickers if t]
    if not tickers:
        return pd.DataFrame(), utc_now_str(), "No tickers provided"
    try:
        data = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        return data, utc_now_str(), None
    except Exception as e:
        return pd.DataFrame(), utc_now_str(), str(e)


@st.cache_data(ttl=1800, show_spinner=False)
def get_info(ticker):
    try:
        info = yf.Ticker(ticker).get_info()
        if not isinstance(info, dict):
            info = {}
        return info, utc_now_str(), None
    except Exception as e:
        return {}, utc_now_str(), str(e)


@st.cache_data(ttl=60, show_spinner=False)
def get_ticker_news_yf(ticker, limit=8):
    try:
        news = yf.Ticker(ticker).news or []
        rows = []
        for item in news[:limit]:
            title = item.get("title") or item.get("content", {}).get("title") or "Untitled"
            publisher = item.get("publisher") or item.get("content", {}).get("provider", {}).get("displayName") or "Yahoo Finance"
            link = item.get("link") or item.get("content", {}).get("canonicalUrl", {}).get("url") or ""
            publish_time = item.get("providerPublishTime")
            if publish_time:
                published = datetime.fromtimestamp(publish_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            else:
                published = "N/A"
            rows.append({
                "Ticker": ticker,
                "Title": title,
                "Source": publisher,
                "Published": published,
                "Link": link,
            })
        return rows, utc_now_str(), None
    except Exception as e:
        return [], utc_now_str(), str(e)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_rss_feeds(portfolio_tickers=None, limit_per_feed=12):
    portfolio_tickers = portfolio_tickers or []
    feeds = dict(RSS_FEEDS)
    if portfolio_tickers:
        quoted = quote_plus(",".join(portfolio_tickers[:20]))
        feeds["Portfolio-specific news"] = [f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quoted}&region=US&lang=en-US"]

    rows = []
    for category, urls in feeds.items():
        for url in urls:
            try:
                parsed = feedparser.parse(url)
                for entry in parsed.entries[:limit_per_feed]:
                    title = getattr(entry, "title", "Untitled")
                    link = getattr(entry, "link", "")
                    source = getattr(parsed.feed, "title", "RSS")
                    published = getattr(entry, "published", None) or getattr(entry, "updated", None) or "N/A"
                    text = f"{title} {getattr(entry, 'summary', '')}".upper()
                    detected = [t for t in portfolio_tickers if re.search(rf"\b{re.escape(t)}\b", text)]
                    relevance = 50
                    if detected:
                        relevance += 35
                    if any(word in text for word in ["EARNINGS", "GUIDANCE", "SEC", "FILING", "BREAKING", "FED", "INFLATION"]):
                        relevance += 10
                    rows.append({
                        "Title": title,
                        "Source": source,
                        "Published": published,
                        "Related ticker": ", ".join(detected) if detected else "N/A",
                        "Category": category,
                        "Relevance score": min(100, relevance),
                        "Why it may matter": explain_headline(title, category, detected),
                        "Link": link,
                    })
            except Exception:
                continue
    df = pd.DataFrame(rows).drop_duplicates(subset=["Title", "Link"], keep="first") if rows else pd.DataFrame()
    if not df.empty:
        df = df.sort_values("Relevance score", ascending=False)
    return df, utc_now_str()


def explain_headline(title, category, detected):
    title_u = title.upper()
    reasons = []
    if detected:
        reasons.append(f"mentions your watchlist/portfolio ticker(s): {', '.join(detected)}")
    if "EARN" in title_u or "GUIDANCE" in title_u:
        reasons.append("may affect revenue, profit expectations, or analyst estimates")
    if "FED" in title_u or "INFLATION" in title_u or "YIELD" in title_u or "RATE" in title_u:
        reasons.append("may affect discount rates and market risk appetite")
    if "AI" in title_u or "CHIP" in title_u or "SEMICONDUCTOR" in title_u:
        reasons.append("may affect AI and semiconductor sentiment")
    if "SEC" in title_u or "FILING" in title_u:
        reasons.append("may point to a new regulatory filing or company disclosure")
    if not reasons:
        reasons.append(f"belongs to the {category} news category")
    return "; ".join(reasons).capitalize() + "."


@st.cache_data(ttl=1800, show_spinner=False)
def get_sec_company_tickers():
    try:
        headers = {"User-Agent": "zero-cost-streamlit-dashboard-v1 contact@example.com"}
        url = "https://www.sec.gov/files/company_tickers.json"
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        rows = []
        for _, item in data.items():
            rows.append({
                "ticker": item.get("ticker", "").upper(),
                "cik": str(item.get("cik_str", "")).zfill(10),
                "title": item.get("title", ""),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=["ticker", "cik", "title"])


@st.cache_data(ttl=1800, show_spinner=False)
def get_recent_sec_filings(ticker, limit=8):
    try:
        mapping = get_sec_company_tickers()
        if mapping.empty:
            return pd.DataFrame(), utc_now_str(), "SEC ticker map unavailable"
        row = mapping[mapping["ticker"] == ticker.upper()]
        if row.empty:
            return pd.DataFrame(), utc_now_str(), "No SEC CIK found. Non-US tickers often do not have SEC filings."
        cik = row.iloc[0]["cik"]
        headers = {"User-Agent": "zero-cost-streamlit-dashboard-v1 contact@example.com"}
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])[:limit]
        dates = recent.get("filingDate", [])[:limit]
        accession = recent.get("accessionNumber", [])[:limit]
        primary_doc = recent.get("primaryDocument", [])[:limit]
        rows = []
        cik_no_zeros = str(int(cik))
        for form, date, acc, doc in zip(forms, dates, accession, primary_doc):
            acc_clean = acc.replace("-", "")
            link = f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{acc_clean}/{doc}"
            rows.append({"Ticker": ticker.upper(), "Form": form, "Filing date": date, "Link": link})
        return pd.DataFrame(rows), utc_now_str(), None
    except Exception as e:
        return pd.DataFrame(), utc_now_str(), str(e)


@st.cache_data(ttl=1800, show_spinner=False)
def get_options_summary(ticker):
    try:
        tk = yf.Ticker(ticker)
        expiries = tk.options
        if not expiries:
            return {}, utc_now_str(), "No options expiries available from yfinance"
        expiry = expiries[0]
        chain = tk.option_chain(expiry)
        calls = chain.calls.copy() if chain.calls is not None else pd.DataFrame()
        puts = chain.puts.copy() if chain.puts is not None else pd.DataFrame()
        call_oi = safe_float(calls.get("openInterest", pd.Series(dtype=float)).fillna(0).sum()) if not calls.empty else np.nan
        put_oi = safe_float(puts.get("openInterest", pd.Series(dtype=float)).fillna(0).sum()) if not puts.empty else np.nan
        call_iv = safe_float(calls.get("impliedVolatility", pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan).mean()) if not calls.empty else np.nan
        put_iv = safe_float(puts.get("impliedVolatility", pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan).mean()) if not puts.empty else np.nan
        return {
            "Nearest expiry": expiry,
            "Call open interest": call_oi,
            "Put open interest": put_oi,
            "Put/Call OI ratio": put_oi / call_oi if call_oi and call_oi > 0 else np.nan,
            "Average call IV": call_iv,
            "Average put IV": put_iv,
        }, utc_now_str(), None
    except Exception as e:
        return {}, utc_now_str(), str(e)

# --------------------------
# Price table calculations
# --------------------------
def slice_ticker_data(data, ticker):
    if data.empty:
        return pd.DataFrame()
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker in data.columns.get_level_values(0):
                return data[ticker].dropna(how="all")
            # yfinance sometimes returns first level as price field for single ticker
            if "Close" in data.columns.get_level_values(0):
                return data.dropna(how="all")
        return data.dropna(how="all")
    except Exception:
        return pd.DataFrame()


def price_metrics_from_history(hist):
    if hist is None or hist.empty:
        return {}
    close_col = "Adj Close" if "Adj Close" in hist.columns else "Close"
    if close_col not in hist.columns:
        return {}
    close = hist[close_col].dropna()
    volume = hist["Volume"].dropna() if "Volume" in hist.columns else pd.Series(dtype=float)
    if close.empty:
        return {}
    price = safe_float(close.iloc[-1])
    prev = safe_float(close.iloc[-2]) if len(close) >= 2 else np.nan
    daily = ((price / prev) - 1) * 100 if prev and not np.isnan(prev) else np.nan

    def ret(days):
        if len(close) <= days:
            return np.nan
        base = safe_float(close.iloc[-days-1])
        return ((price / base) - 1) * 100 if base and not np.isnan(base) else np.nan

    vol = safe_float(volume.iloc[-1]) if not volume.empty else np.nan
    avg_vol = safe_float(volume.tail(20).mean()) if len(volume) >= 5 else np.nan
    rel_vol = vol / avg_vol if avg_vol and avg_vol > 0 else np.nan
    ma20 = safe_float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else np.nan
    ma50 = safe_float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else np.nan
    ma200 = safe_float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else np.nan
    max_drawdown = calc_max_drawdown(close)
    return {
        "Price": price,
        "Daily change %": daily,
        "5-day return": ret(5),
        "20-day return": ret(20),
        "1-month return": ret(21),
        "3-month return": ret(63),
        "6-month return": ret(126),
        "1-year return": ret(252),
        "Volume": vol,
        "Relative volume": rel_vol,
        "MA20": ma20,
        "MA50": ma50,
        "MA200": ma200,
        "Above MA50": price > ma50 if not np.isnan(ma50) else None,
        "Above MA200": price > ma200 if not np.isnan(ma200) else None,
        "Max drawdown estimate": max_drawdown,
    }


def calc_max_drawdown(close):
    try:
        close = pd.Series(close).dropna()
        if close.empty:
            return np.nan
        rolling_max = close.cummax()
        drawdown = (close / rolling_max - 1) * 100
        return safe_float(drawdown.min())
    except Exception:
        return np.nan


def build_market_table(tickers):
    data, updated, err = download_history(tickers, period="3mo", interval="1d")
    rows = []
    for t in tickers:
        hist = slice_ticker_data(data, t)
        m = price_metrics_from_history(hist)
        rows.append({
            "Ticker": t,
            "Price": m.get("Price", np.nan),
            "Daily change %": m.get("Daily change %", np.nan),
            "5-day change %": m.get("5-day return", np.nan),
            "1-month change %": m.get("1-month return", np.nan),
            "Volume vs avg": m.get("Relative volume", np.nan),
            "Last updated": updated,
        })
    return pd.DataFrame(rows), updated, err


def build_ticker_rows(tickers, include_fundamentals=True, include_news=False, include_filings=False):
    data, updated, err = download_history(tickers, period="1y", interval="1d")
    rows = []
    for t in tickers:
        hist = slice_ticker_data(data, t)
        m = price_metrics_from_history(hist)
        info = {}
        info_updated = "N/A"
        if include_fundamentals:
            info, info_updated, _ = get_info(t)
        next_earnings = info.get("earningsDate") or info.get("nextEarningsDate") or None
        row = {
            "Ticker": t,
            "Company name": info.get("shortName") or info.get("longName") or "N/A",
            "Sector": info.get("sector") or "N/A",
            "Price": m.get("Price", np.nan),
            "Last updated": updated,
            "Daily change %": m.get("Daily change %", np.nan),
            "5-day return": m.get("5-day return", np.nan),
            "20-day return": m.get("20-day return", np.nan),
            "3-month return": m.get("3-month return", np.nan),
            "6-month return": m.get("6-month return", np.nan),
            "1-year return": m.get("1-year return", np.nan),
            "Relative volume": m.get("Relative volume", np.nan),
            "Market cap": info.get("marketCap", np.nan),
            "Beta": info.get("beta", np.nan),
            "P/E": info.get("trailingPE", np.nan),
            "Forward P/E": info.get("forwardPE", np.nan),
            "PEG": info.get("pegRatio", np.nan),
            "ROE": safe_float(info.get("returnOnEquity", np.nan)) * 100 if not np.isnan(safe_float(info.get("returnOnEquity", np.nan))) else np.nan,
            "Revenue growth": safe_float(info.get("revenueGrowth", np.nan)) * 100 if not np.isnan(safe_float(info.get("revenueGrowth", np.nan))) else np.nan,
            "Dividend yield": safe_float(info.get("dividendYield", np.nan)) * 100 if not np.isnan(safe_float(info.get("dividendYield", np.nan))) else np.nan,
            "Debt-to-equity": info.get("debtToEquity", np.nan),
            "Free cash flow": info.get("freeCashflow", np.nan),
            "Short % float": safe_float(info.get("shortPercentOfFloat", np.nan)) * 100 if not np.isnan(safe_float(info.get("shortPercentOfFloat", np.nan))) else np.nan,
            "Earnings date / days away": days_until(next_earnings),
            "MA20": m.get("MA20", np.nan),
            "MA50": m.get("MA50", np.nan),
            "MA200": m.get("MA200", np.nan),
            "Max drawdown estimate": m.get("Max drawdown estimate", np.nan),
            "Fundamentals updated": info_updated,
        }
        flags = build_flags(row)
        row["Alerts / flags"] = "; ".join(flags) if flags else "None detected"
        if include_news:
            news, _, _ = get_ticker_news_yf(t, limit=3)
            row["Latest headlines"] = " | ".join([n.get("Title", "") for n in news[:2]]) if news else "N/A"
            row["Fresh news count"] = len(news)
        if include_filings:
            filings, _, _ = get_recent_sec_filings(t, limit=5)
            row["Recent filing count"] = len(filings) if not filings.empty else 0
        rows.append(row)
    return pd.DataFrame(rows), updated, err


def build_flags(row):
    flags = []
    daily = safe_float(row.get("Daily change %"))
    rel_vol = safe_float(row.get("Relative volume"))
    short_float = safe_float(row.get("Short % float"))
    days = safe_float(row.get("Earnings date / days away"))
    price = safe_float(row.get("Price"))
    ma50 = safe_float(row.get("MA50"))
    ma200 = safe_float(row.get("MA200"))
    if not np.isnan(daily) and abs(daily) >= 5:
        flags.append("Big price move today")
    if not np.isnan(rel_vol) and rel_vol >= 1.8:
        flags.append("Volume spike")
    if not np.isnan(days) and 0 <= days <= 14:
        flags.append("Earnings soon")
    if not np.isnan(short_float) and short_float >= 15:
        flags.append("High short interest")
    if not np.isnan(price) and not np.isnan(ma50):
        flags.append("Above 50-day MA" if price > ma50 else "Below 50-day MA")
    if not np.isnan(price) and not np.isnan(ma200):
        flags.append("Above 200-day MA" if price > ma200 else "Below 200-day MA")
    return flags

# --------------------------
# Scoring models: simple V1 heuristics, not investment advice
# --------------------------
def add_speculative_scores(df):
    if df.empty:
        return df
    out = df.copy()
    out["Momentum score"] = out.apply(lambda r: score_clip(
        40
        + safe_float(r.get("Daily change %", 0)) * 2
        + safe_float(r.get("5-day return", 0)) * 1.5
        + safe_float(r.get("20-day return", 0))
        + (10 if safe_float(r.get("Price")) > safe_float(r.get("MA50")) else 0)
        + (10 if safe_float(r.get("Price")) > safe_float(r.get("MA200")) else 0)
    ), axis=1)
    out["Squeeze score"] = out.apply(lambda r: score_clip(
        safe_float(r.get("Short % float", 0)) * 2.5
        + safe_float(r.get("Relative volume", 0)) * 15
        + min(20, safe_float(r.get("Fresh news count", 0)) * 5)
        + min(20, safe_float(r.get("Recent filing count", 0)) * 5)
    ), axis=1)
    out["Risk score"] = out.apply(lambda r: score_clip(
        30
        + abs(safe_float(r.get("Daily change %", 0))) * 3
        + max(0, safe_float(r.get("Beta", 0)) - 1) * 20
        + safe_float(r.get("Short % float", 0))
        + (15 if safe_float(r.get("Market cap", np.nan)) < 5_000_000_000 else 0)
    ), axis=1)
    out["Overall speculative score"] = out.apply(lambda r: score_clip(
        safe_float(r["Momentum score"]) * 0.45 + safe_float(r["Squeeze score"]) * 0.40 + safe_float(r["Risk score"]) * 0.15
    ), axis=1)
    out["Catalyst summary"] = out.apply(catalyst_summary, axis=1)
    out["Why this was flagged"] = out.apply(lambda r: (
        f"Momentum {r['Momentum score']}/100, squeeze {r['Squeeze score']}/100, risk {r['Risk score']}/100. "
        f"Main flags: {r.get('Alerts / flags', 'None detected')}."
    ), axis=1)
    return out.sort_values("Overall speculative score", ascending=False)


def catalyst_summary(r):
    parts = []
    if safe_float(r.get("Fresh news count")) > 0:
        parts.append(f"{int(safe_float(r.get('Fresh news count')))} recent headline(s)")
    if safe_float(r.get("Recent filing count")) > 0:
        parts.append(f"{int(safe_float(r.get('Recent filing count')))} recent SEC filing(s)")
    d = safe_float(r.get("Earnings date / days away"))
    if not np.isnan(d) and 0 <= d <= 30:
        parts.append(f"earnings in about {int(d)} day(s)")
    if not parts:
        parts.append("price/volume/fundamental signal only; no clear fresh catalyst found")
    return "; ".join(parts)


def add_garp_scores(df):
    if df.empty:
        return df
    out = df.copy()
    out["Quality score"] = out.apply(lambda r: score_clip(
        20
        + max(0, safe_float(r.get("ROE", 0))) * 1.2
        + (20 if safe_float(r.get("Free cash flow", np.nan)) > 0 else 0)
        + max(0, safe_float(r.get("Revenue growth", 0)))
        + (10 if safe_float(r.get("Price")) > safe_float(r.get("MA200")) else 0)
    ), axis=1)
    out["Valuation score"] = out.apply(lambda r: score_clip(
        70
        - max(0, safe_float(r.get("Forward P/E", 25)) - 15) * 2
        - max(0, safe_float(r.get("PEG", 2)) - 1.5) * 10
        + (10 if safe_float(r.get("Forward P/E", np.nan)) < 25 or safe_float(r.get("P/E", np.nan)) < 25 else 0)
    ), axis=1)
    out["Balance sheet score"] = out.apply(lambda r: score_clip(
        75
        - max(0, safe_float(r.get("Debt-to-equity", 100)) - 80) * 0.3
        - max(0, safe_float(r.get("Beta", 1)) - 1.2) * 20
        + (10 if safe_float(r.get("Free cash flow", np.nan)) > 0 else 0)
    ), axis=1)
    out["Overall GARP score"] = out.apply(lambda r: score_clip(
        safe_float(r["Quality score"]) * 0.45 + safe_float(r["Valuation score"]) * 0.30 + safe_float(r["Balance sheet score"]) * 0.25
    ), axis=1)
    out["Plain-English explanation"] = out.apply(lambda r: (
        f"Quality {r['Quality score']}/100, valuation {r['Valuation score']}/100, balance sheet {r['Balance sheet score']}/100. "
        f"This is a simple V1 score using available free fundamentals and price trend data."
    ), axis=1)
    return out.sort_values("Overall GARP score", ascending=False)


def add_safe_scores(df):
    if df.empty:
        return df
    out = df.copy()
    out["Overall safe score"] = out.apply(lambda r: score_clip(
        40
        + (20 if safe_float(r.get("Beta", 2)) <= 1.0 else 0)
        + (15 if safe_float(r.get("Beta", 2)) <= 0.8 else 0)
        + min(20, max(0, safe_float(r.get("Dividend yield", 0))) * 4)
        + (15 if safe_float(r.get("Free cash flow", np.nan)) > 0 else 0)
        + (10 if safe_float(r.get("Market cap", 0)) >= 10_000_000_000 else 0)
        + (10 if str(r.get("Sector", "")) in SECTOR_DEFENSIVE else 0)
        - max(0, safe_float(r.get("Debt-to-equity", 100)) - 80) * 0.2
        + (10 if safe_float(r.get("Price")) > safe_float(r.get("MA200")) else 0)
    ), axis=1)
    out["Plain-English explanation"] = out.apply(lambda r: (
        f"Safe score is based on beta, dividend yield, free cash flow, market cap, sector defensiveness, debt, and long-term trend. "
        f"Available flags: {r.get('Alerts / flags', 'None detected')}."
    ), axis=1)
    return out.sort_values("Overall safe score", ascending=False)

# --------------------------
# Display helpers
# --------------------------
def display_dataframe(df, cols=None, height=520):
    if df is None or df.empty:
        st.info("No data available right now. Free sources may be delayed, unavailable, or rate-limited.")
        return
    show = df.copy()
    if cols:
        show = show[[c for c in cols if c in show.columns]]
    formatters = {}
    for col in show.columns:
        if col in ["Price", "P/E", "Forward P/E", "PEG", "Beta", "Relative volume"]:
            formatters[col] = "{:.2f}"
        elif "%" in col or "return" in col.lower() or "drawdown" in col.lower() or col in ["ROE", "Revenue growth", "Dividend yield", "Short % float"]:
            formatters[col] = "{:.2f}"
        elif col in ["Market cap", "Free cash flow"]:
            show[col] = show[col].apply(fmt_money)
    st.dataframe(show, use_container_width=True, height=height)


def metric_row_from_df(df, tickers):
    cols = st.columns(min(4, len(tickers)))
    for i, t in enumerate(tickers[:4]):
        row = df[df["Ticker"] == t]
        with cols[i % len(cols)]:
            if row.empty:
                st.metric(t, "N/A", "N/A")
            else:
                r = row.iloc[0]
                st.metric(t, fmt_money(r.get("Price")), fmt_pct(r.get("Daily change %")))


def market_regime(df):
    if df.empty:
        return "Mixed/choppy", "Market data is unavailable, so the app cannot confidently classify the regime."
    lookup = {r["Ticker"]: r for _, r in df.iterrows()}
    spy = safe_float(lookup.get("SPY", {}).get("Daily change %", np.nan))
    qqq = safe_float(lookup.get("QQQ", {}).get("Daily change %", np.nan))
    iwm = safe_float(lookup.get("IWM", {}).get("Daily change %", np.nan))
    tlt = safe_float(lookup.get("TLT", {}).get("Daily change %", np.nan))
    vix = safe_float(lookup.get("^VIX", {}).get("Daily change %", np.nan))
    xle = safe_float(lookup.get("XLE", {}).get("Daily change %", np.nan))
    score = 0
    for x in [spy, qqq, iwm]:
        if not np.isnan(x):
            score += 1 if x > 0.3 else -1 if x < -0.3 else 0
    if not np.isnan(vix):
        score += -1 if vix > 5 else 1 if vix < -3 else 0
    if score >= 3:
        return "Risk-on", "Stocks are broadly positive and volatility is not flashing major stress. Growth and higher-beta areas may be getting bid."
    if score <= -3:
        return "Risk-off", "Major equity indexes are weak and/or volatility is rising. The market may be reducing risk."
    if not np.isnan(xle) and not np.isnan(qqq) and xle > qqq + 1:
        return "Defensive rotation", "Energy or defensive areas are outperforming growth, which can happen when investors rotate away from higher-duration assets."
    if not np.isnan(vix) and vix > 8:
        return "High-volatility caution", "Volatility is moving sharply. Position sizing and risk control matter more than usual."
    return "Mixed/choppy", "Signals are not clearly bullish or bearish. Some groups may be working while others are under pressure."


def plot_price_chart(ticker, period="6mo", interval="1d", show_ma=True):
    data, updated, err = download_history([ticker], period=period, interval=interval)
    hist = slice_ticker_data(data, ticker)
    if hist.empty or "Close" not in hist.columns:
        st.info("Chart data unavailable right now.")
        return
    close = hist["Close"].dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, mode="lines", name="Close"))
    if show_ma and interval == "1d":
        for window in [20, 50, 200]:
            if len(close) >= window:
                fig.add_trace(go.Scatter(x=close.index, y=close.rolling(window).mean(), mode="lines", name=f"{window}D MA"))
    fig.update_layout(
        title=f"{ticker} price chart — source: yfinance — last updated: {updated}",
        template="plotly_dark",
        height=430,
        margin=dict(l=20, r=20, t=55, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_volume_chart(ticker, period="6mo"):
    data, updated, err = download_history([ticker], period=period, interval="1d")
    hist = slice_ticker_data(data, ticker)
    if hist.empty or "Volume" not in hist.columns:
        st.info("Volume data unavailable right now.")
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], name="Volume"))
    fig.update_layout(
        title=f"{ticker} volume — source: yfinance — last updated: {updated}",
        template="plotly_dark",
        height=300,
        margin=dict(l=20, r=20, t=55, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Sidebar controls
# --------------------------
st.sidebar.title("📱 iPad Market Dashboard V1")
st.sidebar.caption("Designed for Streamlit Community Cloud + Safari on iPad.")

refresh_choice = st.sidebar.radio(
    "Refresh mode",
    ["15 seconds", "30 seconds", "60 seconds", "Manual only"],
    index=2,
    help="15 seconds can be too aggressive for free data sources and may cause rate limits.",
)
if refresh_choice != "Manual only":
    seconds = int(refresh_choice.split()[0])
    if seconds == 15:
        st.sidebar.warning("15-second refresh may trigger free-source rate limits. Use 60 seconds if data starts failing.")
    st.markdown(f"<meta http-equiv='refresh' content='{seconds}'>", unsafe_allow_html=True)

if st.sidebar.button("Refresh now"):
    st.cache_data.clear()
    st.rerun()

if "portfolio_text" not in st.session_state:
    st.session_state.portfolio_text = DEFAULT_PORTFOLIO

portfolio_text = st.sidebar.text_area(
    "Portfolio / watchlist tickers",
    value=st.session_state.portfolio_text,
    height=90,
    help="Type tickers separated by commas. This is remembered only during your current Streamlit session in Version 1.",
)
st.session_state.portfolio_text = portfolio_text
portfolio_tickers = normalize_tickers(portfolio_text, max_count=30)

max_universe = st.sidebar.slider("Screener universe size", 20, min(100, len(DEFAULT_UNIVERSE)), 60, step=10)
screener_universe = DEFAULT_UNIVERSE[:max_universe]
selected_ticker = st.sidebar.text_input("Stock detail ticker", value=(portfolio_tickers[0] if portfolio_tickers else "NVDA")).upper().strip()

st.sidebar.markdown("---")
st.sidebar.caption("Data sources: yfinance, RSS feeds, SEC EDGAR. Optional Finnhub support is left as a future add-on.")
st.sidebar.caption(f"Page rendered: {utc_now_str()}")

# --------------------------
# App header
# --------------------------
st.title("📈 Zero-Cost Equity Research Dashboard — Version 1")
st.markdown(f"<div class='warning-box'><b>Honesty note:</b> {DISCLAIMER}</div>", unsafe_allow_html=True)

with st.expander("What this Version 1 app can and cannot do", expanded=False):
    st.write(
        "This first version is built to run on Streamlit Community Cloud from GitHub. It focuses on prices, charts, "
        "portfolio monitoring, RSS headlines, simple screeners, basic fundamentals, options summaries where yfinance provides them, "
        "and SEC filings for U.S.-listed companies. It does not store a permanent portfolio database yet, and it does not provide "
        "paid institutional real-time data."
    )

# --------------------------
# Main tabs
# --------------------------
tabs = st.tabs([
    "🏠 Home / Market",
    "💼 Portfolio",
    "📰 Live News",
    "🚀 Speculative Screener",
    "🏗️ GARP Screener",
    "🛡️ Safe Screener",
    "🔎 Stock Detail",
    "🎓 Beginner Guide",
])

# Home / Market
with tabs[0]:
    st.subheader("Home / Market Command Center")
    market_df, market_updated, market_err = build_market_table(MARKET_TICKERS)
    if market_err:
        st.warning(f"Market data warning: {market_err}")
    label, explanation = market_regime(market_df)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Market regime", label)
    with c2:
        st.markdown(f"<div class='risk-box'>{explanation}</div>", unsafe_allow_html=True)
    metric_row_from_df(market_df, ["SPY", "QQQ", "IWM", "^VIX"])
    st.caption(f"Source: yfinance. Last updated: {market_updated}")
    display_dataframe(market_df, height=470)

# Portfolio
with tabs[1]:
    st.subheader("Portfolio Dashboard")
    st.caption("Version 1 uses the sidebar text box and Streamlit session state. It remembers your tickers during the current session only.")
    if not portfolio_tickers:
        st.info("Add portfolio tickers in the sidebar.")
    else:
        portfolio_df, updated, err = build_ticker_rows(portfolio_tickers, include_fundamentals=True, include_news=True, include_filings=True)
        if err:
            st.warning(f"Portfolio data warning: {err}")
        st.caption(f"Prices source: yfinance. Fundamentals cache: 30 minutes. Last price refresh: {updated}")
        metric_row_from_df(portfolio_df, portfolio_tickers[:4])
        st.markdown("### What changed today")
        movers = portfolio_df.copy()
        if not movers.empty:
            movers["Abs move"] = movers["Daily change %"].apply(lambda x: abs(safe_float(x)) if not np.isnan(safe_float(x)) else -1)
            top = movers.sort_values("Abs move", ascending=False).head(5)
            summary_lines = []
            for _, r in top.iterrows():
                summary_lines.append(
                    f"**{r['Ticker']}** moved **{fmt_pct(r.get('Daily change %'))}** today. "
                    f"Relative volume is **{fmt_num(r.get('Relative volume'))}x**. Flags: {r.get('Alerts / flags', 'None detected')}."
                )
            st.markdown("\n\n".join(summary_lines))
        cols = [
            "Ticker", "Company name", "Price", "Daily change %", "5-day return", "20-day return", "3-month return",
            "Relative volume", "Market cap", "Sector", "Beta", "P/E", "Forward P/E", "Dividend yield",
            "Debt-to-equity", "Free cash flow", "Short % float", "Earnings date / days away", "Alerts / flags",
            "Latest headlines", "Recent filing count", "Last updated"
        ]
        display_dataframe(portfolio_df, cols=cols, height=560)

# News
with tabs[2]:
    st.subheader("Live News Dashboard")
    news_df, news_updated = fetch_rss_feeds(portfolio_tickers=portfolio_tickers, limit_per_feed=12)
    st.caption(f"Sources: RSS feeds including Yahoo Finance and Investing.com where available. Last updated: {news_updated}")
    if news_df.empty:
        st.info("No headlines available right now.")
    else:
        categories = ["All"] + sorted(news_df["Category"].dropna().unique().tolist())
        selected_cat = st.selectbox("Filter category", categories)
        filtered = news_df if selected_cat == "All" else news_df[news_df["Category"] == selected_cat]
        display_dataframe(filtered, height=620)

# Speculative screener
with tabs[3]:
    st.subheader("Speculative / Tactical / Squeeze / Catalyst Screener")
    st.markdown("<div class='warning-box'>This screener is high-risk and does not mean the stock is safe or guaranteed to move up.</div>", unsafe_allow_html=True)
    st.caption("Sources: yfinance prices/fundamentals/news where available; SEC EDGAR recent filings for U.S. companies. Fast data cache 60 seconds; fundamentals/filings cache 30 minutes.")
    base_df, updated, err = build_ticker_rows(screener_universe, include_fundamentals=True, include_news=True, include_filings=True)
    spec_df = add_speculative_scores(base_df)
    if err:
        st.warning(f"Screener warning: {err}")
    cols = [
        "Ticker", "Company name", "Price", "Last updated", "Daily change %", "5-day return", "20-day return",
        "Relative volume", "Short % float", "Earnings date / days away", "Fresh news count", "Recent filing count",
        "Catalyst summary", "Squeeze score", "Momentum score", "Risk score", "Overall speculative score", "Why this was flagged"
    ]
    display_dataframe(spec_df, cols=cols, height=650)

# GARP screener
with tabs[4]:
    st.subheader("Quality / GARP / Mid-to-Long-Term Screener")
    st.caption("Purpose: find better-quality stocks that may be reasonable 1–3+ year watchlist candidates. This is a simple V1 heuristic, not a recommendation.")
    base_df, updated, err = build_ticker_rows(screener_universe, include_fundamentals=True, include_news=False, include_filings=False)
    garp_df = add_garp_scores(base_df)
    cols = [
        "Ticker", "Company name", "Sector", "Price", "Last updated", "Market cap", "P/E", "Forward P/E", "PEG",
        "ROE", "Revenue growth", "Free cash flow", "Debt-to-equity", "Beta", "6-month return", "1-year return",
        "Quality score", "Valuation score", "Balance sheet score", "Overall GARP score", "Plain-English explanation"
    ]
    display_dataframe(garp_df, cols=cols, height=650)

# Safe screener
with tabs[5]:
    st.subheader("Safe / Set-and-Forget / Low-Volatility Screener")
    st.caption("Purpose: find lower-volatility defensive watchlist candidates. This is still stock-market risk, not guaranteed safety.")
    base_df, updated, err = build_ticker_rows(screener_universe, include_fundamentals=True, include_news=False, include_filings=False)
    safe_df = add_safe_scores(base_df)
    cols = [
        "Ticker", "Company name", "Sector", "Price", "Last updated", "Market cap", "Dividend yield", "Beta", "P/E",
        "Debt-to-equity", "Free cash flow", "1-year return", "Max drawdown estimate", "Overall safe score", "Plain-English explanation"
    ]
    display_dataframe(safe_df, cols=cols, height=650)

# Stock detail
with tabs[6]:
    st.subheader("Stock Detail / Deep Dive")
    if not selected_ticker:
        st.info("Enter a ticker in the sidebar.")
    else:
        info, info_updated, info_err = get_info(selected_ticker)
        rows_df, updated, err = build_ticker_rows([selected_ticker], include_fundamentals=True, include_news=True, include_filings=True)
        r = rows_df.iloc[0].to_dict() if not rows_df.empty else {}
        st.caption(f"Sources: yfinance, Yahoo Finance news where available, SEC EDGAR for U.S. filings. Price updated: {updated}. Fundamentals updated: {info_updated}.")
        if info_err:
            st.warning(f"Fundamentals warning: {info_err}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", fmt_money(r.get("Price")), fmt_pct(r.get("Daily change %")))
        c2.metric("Relative volume", fmt_num(r.get("Relative volume")), "x avg")
        c3.metric("Market cap", fmt_money(r.get("Market cap")))
        c4.metric("Beta", fmt_num(r.get("Beta")))

        chart_period = st.radio("Chart range", ["1d", "1mo", "6mo", "1y", "5y"], horizontal=True, index=2)
        if chart_period == "1d":
            plot_price_chart(selected_ticker, period="1d", interval="5m", show_ma=False)
        else:
            plot_price_chart(selected_ticker, period=chart_period, interval="1d", show_ma=True)
        plot_volume_chart(selected_ticker, period="6mo")

        st.markdown("### Key fundamentals")
        fundamental_cols = [
            "Ticker", "Company name", "Sector", "Price", "P/E", "Forward P/E", "PEG", "ROE", "Revenue growth",
            "Free cash flow", "Debt-to-equity", "Dividend yield", "Short % float", "Earnings date / days away", "Alerts / flags"
        ]
        display_dataframe(rows_df, cols=fundamental_cols, height=180)

        st.markdown("### Business explanation")
        business = info.get("longBusinessSummary") or "N/A"
        st.write(business)

        analyst_target = info.get("targetMeanPrice", np.nan)
        recommendation = info.get("recommendationKey", "N/A")
        st.markdown("### Analyst / market fields available from free data")
        st.write(f"Analyst target mean price: **{fmt_money(analyst_target)}** | Recommendation key: **{recommendation}**")

        st.markdown("### Options summary")
        opt, opt_updated, opt_err = get_options_summary(selected_ticker)
        if opt:
            st.caption(f"Source: yfinance options chain. Last updated: {opt_updated}")
            st.dataframe(pd.DataFrame([opt]), use_container_width=True)
        else:
            st.info(f"Options summary unavailable: {opt_err or 'N/A'}")

        st.markdown("### Latest news")
        news, news_updated, news_err = get_ticker_news_yf(selected_ticker, limit=8)
        if news:
            st.caption(f"Source: yfinance ticker news. Last updated: {news_updated}")
            st.dataframe(pd.DataFrame(news), use_container_width=True, height=270)
        else:
            st.info(f"No ticker news available: {news_err or 'N/A'}")

        st.markdown("### Recent SEC filings")
        filings, filings_updated, filings_err = get_recent_sec_filings(selected_ticker, limit=10)
        if not filings.empty:
            st.caption(f"Source: SEC EDGAR submissions API. Last updated: {filings_updated}")
            st.dataframe(filings, use_container_width=True, height=300)
        else:
            st.info(f"SEC filings unavailable or not applicable: {filings_err or 'N/A'}")

        st.markdown("### Risk / reward summary")
        up_reasons = []
        down_reasons = []
        if safe_float(r.get("Price")) > safe_float(r.get("MA50")):
            up_reasons.append("price is above the 50-day moving average")
        else:
            down_reasons.append("price is below the 50-day moving average")
        if safe_float(r.get("Revenue growth")) > 0:
            up_reasons.append("reported revenue growth is positive where available")
        if safe_float(r.get("Free cash flow")) > 0:
            up_reasons.append("free cash flow appears positive where available")
        if safe_float(r.get("Debt-to-equity")) > 150:
            down_reasons.append("debt-to-equity appears elevated where available")
        if safe_float(r.get("Short % float")) > 15:
            down_reasons.append("short interest appears high, which can increase volatility")
        if not up_reasons:
            up_reasons.append("a positive catalyst such as earnings, guidance, sector momentum, or major news would be needed")
        if not down_reasons:
            down_reasons.append("valuation, earnings disappointment, sector weakness, or market risk could still pressure the stock")
        st.write("**What could make it go up:** " + "; ".join(up_reasons) + ".")
        st.write("**What could make it go down:** " + "; ".join(down_reasons) + ".")
        st.write("**Missing or unreliable data:** Free-source short interest, options data, analyst targets, and fundamentals can be stale, incomplete, or unavailable. Treat N/A as unknown, not as zero.")

# Beginner guide
with tabs[7]:
    st.subheader("Beginner Education Layer")
    explanations = {
        "Short interest": "The percentage of a company’s tradable shares that investors have borrowed and sold short. High short interest can mean bearish sentiment or potential squeeze risk.",
        "Short squeeze": "A sharp move up caused when short sellers rush to buy shares back, adding more buying pressure.",
        "Relative volume": "Today’s volume compared with normal volume. A value near 2.0 means trading is roughly twice normal activity.",
        "Market cap": "The company’s total stock-market value: share price multiplied by shares outstanding.",
        "Beta": "A rough measure of volatility compared with the market. Beta above 1 usually means more volatile than the market.",
        "P/E ratio": "Price-to-earnings ratio. It compares stock price to trailing earnings. Lower is not always better; growth and quality matter.",
        "Forward P/E": "Price compared with expected future earnings. This depends on analyst estimates and can be wrong.",
        "PEG": "P/E ratio adjusted for growth. A lower PEG can suggest a better growth-adjusted valuation, but data quality varies.",
        "ROE": "Return on equity. It estimates how efficiently a company generates profit from shareholder equity.",
        "Free cash flow": "Cash left after operating expenses and capital spending. Positive free cash flow is often a quality signal.",
        "Debt-to-equity": "A leverage measure comparing debt to shareholder equity. Very high values may increase financial risk.",
        "Dividend yield": "Annual dividends divided by stock price. A high yield can be attractive, but can also signal risk if the dividend is unsustainable.",
        "Moving averages": "Average prices over a period, such as 20, 50, or 200 days. Traders use them to judge trend direction.",
        "Support and resistance": "Support is an area where buyers may appear. Resistance is an area where sellers may appear. These are estimates, not guarantees.",
        "Earnings catalyst": "A scheduled earnings report can move a stock sharply because new numbers and guidance reset expectations.",
        "SEC 8-K": "A U.S. company filing used for important current events such as major agreements, leadership changes, or material updates.",
        "Form 4 insider filing": "A filing showing insider purchases or sales. It can matter, but context is important.",
        "13D/G ownership filing": "A filing showing significant ownership stakes, often by large investors or activists.",
        "Options open interest": "The number of open option contracts. High open interest can show where traders are concentrated.",
        "Implied volatility": "The options market’s estimate of expected future volatility. Higher IV means options are pricing larger moves.",
    }
    for term, text in explanations.items():
        with st.expander(term):
            st.write(text)

st.markdown("---")
st.caption(f"{DISCLAIMER} Rendered at {utc_now_str()}.")
