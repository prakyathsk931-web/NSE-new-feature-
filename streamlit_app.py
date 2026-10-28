"""
NSE Stock Analysis Dashboard — Streamlit frontend (100% free)
=============================================================
This app does NOT calculate anything itself. It calls the FastAPI service you
already deployed on Render and draws the results with Plotly.

Setup:
  1. Put this file in the ROOT of a GitHub repo as  streamlit_app.py
  2. Put requirements.txt (below) next to it
  3. On share.streamlit.io -> Create app -> pick the repo -> Main file path: streamlit_app.py
  4. Advanced settings -> Secrets -> add:  API_BASE_URL = "https://YOUR-APP.onrender.com"
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="NSE Stock Analysis", page_icon="📈", layout="wide")

DEFAULT_URL = "https://YOUR-APP.onrender.com"   # <-- change this, or use Streamlit secrets


def default_base_url():
    try:
        if "API_BASE_URL" in st.secrets:
            return st.secrets["API_BASE_URL"]
    except Exception:
        pass
    return os.environ.get("API_BASE_URL", DEFAULT_URL)


# ─────────────────────────── API helper ───────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _fetch(base_url, path, params_tuple):
    r = requests.get(f"{base_url.rstrip('/')}{path}", params=dict(params_tuple), timeout=120)
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json()


def api(path, **params):
    """Returns the JSON, or None (after showing an error) if the call failed."""
    try:
        with st.spinner("Loading... (the free server may need up to a minute to wake up)"):
            return _fetch(BASE_URL, path, tuple(sorted(params.items())))
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return None


def fmt(v, suffix="", prefix=""):
    return "N/A" if v is None else f"{prefix}{v:,}{suffix}"


def signal_color(label):
    return {"Bullish": "🟢", "Mildly Bullish": "🟢", "Neutral": "⚪",
            "Mildly Bearish": "🔴", "Bearish": "🔴"}.get(label, "⚪")


# ─────────────────────────── Sidebar ──────────────────────────────────────

st.sidebar.title("📈 NSE Analyzer")
BASE_URL = st.sidebar.text_input("API URL (Render)", value=default_base_url())
PAGE = st.sidebar.radio(
    "Go to",
    ["Market Overview", "Stock Analysis", "Risk", "Compare", "Backtest",
     "Valuation", "News", "Forecast", "Watchlist"],
)
st.sidebar.caption("Educational project — not investment advice.")

SINGLE_STOCK_PAGES = {"Stock Analysis", "Risk", "Backtest", "Valuation", "News", "Forecast"}
symbol = "RELIANCE"
if PAGE in SINGLE_STOCK_PAGES:
    symbol = st.sidebar.text_input("NSE symbol", value="RELIANCE").upper().strip().replace(".NS", "")

st.title(PAGE)


# ─────────────────────────── Pages ────────────────────────────────────────

def page_market():
    data = api("/market-overview")
    if not data:
        return
    if data.get("market_mood"):
        st.subheader(f"Market mood: {data['market_mood']}")
    cols = st.columns(3)
    for i, idx in enumerate(data["indices"]):
        with cols[i % 3]:
            if "error" in idx:
                st.metric(idx["name"], "N/A")
            else:
                st.metric(idx["name"], f"{idx['price']:,}", f"{idx['change_pct']}% today")
                st.caption(f"1-month: {idx['change_1m_pct']}%  |  as of {idx['as_of']}")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("🟢 Top gainers")
        st.dataframe(pd.DataFrame(data["top_gainers"]), hide_index=True, width="stretch")
    with right:
        st.subheader("🔴 Top losers")
        st.dataframe(pd.DataFrame(data["top_losers"]), hide_index=True, width="stretch")
    st.caption(f"Advancers: {data['advancers']}  |  Decliners: {data['decliners']}  "
               f"(basket of {data['basket_size']} large caps)")


def page_analysis():
    months = st.sidebar.slider("Chart months", 1, 24, 6)
    data = api(f"/analyze/{symbol}")
    if not data:
        return
    t, f, sig = data["technical"], data["fundamentals"], data["signal_summary"]

    if data.get("corporate_action_note"):
        st.warning(data["corporate_action_note"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", fmt(t["current_price"], prefix="₹"))
    c2.metric("RSI (14)", fmt(t["rsi_14"]))
    c3.metric("Trend (50/200 MA)", (t["ma_trend"] or "N/A").title())
    c4.metric("Signal", f"{signal_color(sig['label'])} {sig['label']}", f"score {sig['score']}/{sig['max_score']}")

    with st.expander("Why this signal? (indicator votes)"):
        for v in sig["votes"]:
            icon = "➕" if v["score"] > 0 else "➖" if v["score"] < 0 else "⚪"
            st.write(f"{icon} **{v['indicator']}** — {v['reason']}")

    chart = api(f"/chart/{symbol}", months=months)
    if chart:
        df = pd.DataFrame(chart["data"])
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.22, 0.23],
                            vertical_spacing=0.03, subplot_titles=("Price", "RSI", "MACD"))
        fig.add_trace(go.Candlestick(x=df["date"], open=df["open"], high=df["high"], low=df["low"],
                                     close=df["close"], name="Price"), row=1, col=1)
        for col, name in [("ma_50", "MA 50"), ("ma_200", "MA 200"), ("bb_upper", "BB upper"), ("bb_lower", "BB lower")]:
            fig.add_trace(go.Scatter(x=df["date"], y=df[col], name=name, mode="lines",
                                     line=dict(width=1, dash="dot" if col.startswith("bb") else "solid")),
                          row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["rsi"], name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["macd_line"], name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["macd_signal"], name="Signal"), row=3, col=1)
        fig.update_layout(height=750, xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Fundamentals")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Earnings growth YoY", fmt(f["yoy_earnings_growth_pct"], "%"))
    g2.metric("Revenue growth YoY", fmt(f["yoy_revenue_growth_pct"], "%"))
    g3.metric("Net margin", fmt(f["net_margin_pct"], "%"))
    g4.metric("ROE (approx.)", fmt(f["roe_pct"], "%"))
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Operating margin", fmt(f["operating_margin_pct"], "%"))
    h2.metric("Debt / Equity", fmt(f["debt_to_equity"]))
    h3.metric("Market cap (₹ Cr)", fmt(f["market_cap_cr"]))
    h4.metric("MACD histogram", fmt(t["macd_histogram"]))


def page_risk():
    years = st.sidebar.slider("Years of history", 1, 5, 2)
    d = api(f"/risk/{symbol}", years=years)
    if not d:
        return
    st.subheader(f"Risk level: {d['risk_level']}")
    a = st.columns(4)
    a[0].metric("CAGR", fmt(d["annualised_return_cagr_pct"], "%"))
    a[1].metric("Volatility (annual)", fmt(d["annualised_volatility_pct"], "%"))
    a[2].metric("Sharpe ratio", fmt(d["sharpe_ratio"]))
    a[3].metric("Sortino ratio", fmt(d["sortino_ratio"]))
    b = st.columns(4)
    b[0].metric("Max drawdown", fmt(d["max_drawdown_pct"], "%"))
    b[1].metric("Beta vs NIFTY", fmt(d["beta_vs_nifty"]))
    b[2].metric("Correlation vs NIFTY", fmt(d["correlation_vs_nifty"]))
    b[3].metric("Positive days", fmt(d["positive_days_pct"], "%"))
    c = st.columns(4)
    c[0].metric("VaR 95% (1 day)", fmt(d["var_95_daily_pct"], "%"))
    c[1].metric("VaR 99% (1 day)", fmt(d["var_99_daily_pct"], "%"))
    c[2].metric("Best day", fmt(d["best_day_pct"], "%"))
    c[3].metric("Worst day", fmt(d["worst_day_pct"], "%"))
    st.caption(f"Worst fall: peak {d['max_drawdown_peak_date']} → trough {d['max_drawdown_trough_date']}. "
               f"Sharpe uses a {d['risk_free_rate_pct']:.0f}% risk-free rate.")
    with st.expander("What do these mean?"):
        st.markdown(
            "- **Volatility**: how much the price swings in a year.\n"
            "- **Sharpe / Sortino**: return earned per unit of risk (Sortino counts only downside risk).\n"
            "- **Max drawdown**: biggest fall from a peak.\n"
            "- **VaR 95%**: on 95% of days the loss should be smaller than this.\n"
            "- **Beta**: above 1 moves more than the NIFTY, below 1 moves less."
        )


def page_compare():
    raw = st.text_input("Symbols (2-6, comma separated)", value="TCS,INFY,WIPRO")
    years = st.slider("Years", 1, 5, 1)
    d = api("/compare", symbols=raw, years=years)
    if not d:
        return
    for w in d.get("warnings", []):
        st.warning(w)
    st.success(f"Best: {d['best_performer']}   |   Worst: {d['worst_performer']}")
    series = pd.DataFrame(d["rebased_series"]).set_index("date")
    st.plotly_chart(px.line(series, title="Performance (start = 100)"), width="stretch")
    st.dataframe(pd.DataFrame(d["summary"]), hide_index=True, width="stretch")
    corr = pd.DataFrame(d["correlation_matrix"])
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn", zmin=-1, zmax=1,
                              title="Correlation of daily returns"), width="stretch")
    st.caption("Low correlation between stocks = better diversification.")


def page_backtest():
    c = st.columns(4)
    short = c[0].number_input("Short MA", 5, 100, 50)
    long = c[1].number_input("Long MA", 10, 250, 200)
    years = c[2].slider("Years", 1, 5, 3)
    capital = c[3].number_input("Capital (₹)", 10000, 10000000, 100000, step=10000)
    d = api(f"/backtest/{symbol}", short=int(short), long=int(long), years=years, capital=float(capital))
    if not d:
        return
    st.info(f"{d['verdict']}  —  {d['strategy_name']}")
    a, b = st.columns(2)
    with a:
        st.subheader("Strategy")
        st.metric("Final value", fmt(d["strategy"]["final_value"], prefix="₹"), f"{d['strategy']['total_return_pct']}%")
        st.write(f"CAGR {d['strategy']['cagr_pct']}%  |  Max drawdown {d['strategy']['max_drawdown_pct']}%")
    with b:
        st.subheader("Buy & hold")
        st.metric("Final value", fmt(d["buy_and_hold"]["final_value"], prefix="₹"), f"{d['buy_and_hold']['total_return_pct']}%")
        st.write(f"CAGR {d['buy_and_hold']['cagr_pct']}%  |  Max drawdown {d['buy_and_hold']['max_drawdown_pct']}%")
    st.write(f"Trades: **{d['number_of_trades']}**  |  Win rate: **{fmt(d['win_rate_pct'], '%')}**  |  "
             f"Time in market: **{d['time_in_market_pct']}%**")
    curve = pd.DataFrame(d["equity_curve"]).set_index("date")[["strategy", "buy_and_hold"]]
    st.plotly_chart(px.line(curve, title="Growth of your capital"), width="stretch")
    st.caption(d["disclaimer"])


def page_valuation():
    d = api(f"/valuation/{symbol}")
    if not d:
        return
    a = st.columns(4)
    a[0].metric("Price", fmt(d["price"], prefix="₹"))
    a[1].metric("P/E (TTM)", fmt(d["pe_ratio_ttm"]))
    a[2].metric("P/B", fmt(d["pb_ratio"]))
    a[3].metric("EPS (TTM)", fmt(d["eps_ttm"], prefix="₹"))
    b = st.columns(4)
    b[0].metric("Market cap (₹ Cr)", fmt(d["market_cap_cr"]))
    b[1].metric("Dividend yield", fmt(d["dividend_yield_pct"], "%"))
    b[2].metric("52W high", fmt(d["week52_high"], prefix="₹"))
    b[3].metric("52W low", fmt(d["week52_low"], prefix="₹"))
    if d["pe_view"]:
        st.info(d["pe_view"])
    if d["week52_high"] and d["week52_low"] and d["week52_high"] > d["week52_low"]:
        pos = (d["price"] - d["week52_low"]) / (d["week52_high"] - d["week52_low"])
        st.write("Position in 52-week range")
        st.progress(min(max(pos, 0.0), 1.0))
    if d["recent_dividends"]:
        ddf = pd.DataFrame(d["recent_dividends"])
        st.plotly_chart(px.bar(ddf, x="date", y="dividend_per_share", title="Recent dividends (₹ per share)"),
                        width="stretch")
    st.caption(d["note"])


def page_news():
    d = api(f"/news/{symbol}", limit=12)
    if not d:
        return
    if not d["articles"]:
        st.info("No news available right now (Yahoo may be limiting the server). Try again later.")
        return
    c = st.columns(4)
    c[0].metric("Overall", d["overall_sentiment"])
    c[1].metric("Positive", d["positive"])
    c[2].metric("Neutral", d["neutral"])
    c[3].metric("Negative", d["negative"])
    icons = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
    for a in d["articles"]:
        title = f"[{a['title']}]({a['link']})" if a.get("link") else a["title"]
        st.markdown(f"{icons[a['sentiment']]} {title}")
        st.caption(f"{a.get('publisher') or ''}  {a.get('published') or ''}")
    st.caption(d["note"])


def page_forecast():
    days = st.slider("Days ahead", 5, 90, 30)
    d = api(f"/forecast/{symbol}", days=days)
    if not d:
        return
    a = st.columns(4)
    a[0].metric("Current", fmt(d["current_price"], prefix="₹"))
    a[1].metric("Median", fmt(d["median_price"], prefix="₹"))
    a[2].metric("Best case (95%)", fmt(d["best_case_p95"], prefix="₹"))
    a[3].metric("Worst case (5%)", fmt(d["worst_case_p5"], prefix="₹"))
    b = st.columns(3)
    b[0].metric("Chance of gain", fmt(d["probability_of_gain_pct"], "%"))
    b[1].metric("Chance of +10%", fmt(d["probability_of_10pct_gain_pct"], "%"))
    b[2].metric("Chance of -10%", fmt(d["probability_of_10pct_loss_pct"], "%"))
    df = pd.DataFrame(d["bands"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["p95"], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df["date"], y=df["p5"], fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(100,150,250,0.2)", name="90% range"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["p75"], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df["date"], y=df["p25"], fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(100,150,250,0.4)", name="50% range"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["median"], name="Median", line=dict(width=2)))
    fig.add_hline(y=d["current_price"], line_dash="dash", annotation_text="Current price")
    fig.update_layout(title=f"{d['simulations']} simulated price paths", height=500)
    st.plotly_chart(fig, width="stretch")
    st.caption(d["disclaimer"])


def page_watchlist():
    raw = st.text_input("Symbols (max 10, comma separated)", value="RELIANCE,TCS,INFY,HDFCBANK,SBIN")
    d = api("/watchlist", symbols=raw)
    if not d:
        return
    df = pd.DataFrame(d["stocks"])
    st.dataframe(df, hide_index=True, width="stretch")


{
    "Market Overview": page_market,
    "Stock Analysis": page_analysis,
    "Risk": page_risk,
    "Compare": page_compare,
    "Backtest": page_backtest,
    "Valuation": page_valuation,
    "News": page_news,
    "Forecast": page_forecast,
    "Watchlist": page_watchlist,
}[PAGE]()
