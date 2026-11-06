import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NSE Terminal Pro",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0b0d12;
    color: #f5f5f5;
}
.block-container {
    padding-top: 1.2rem;
    max-width: 1550px;
}
[data-testid="stMetric"] {
    background: #171b24;
    border: 1px solid #303746;
    padding: 13px;
    border-radius: 12px;
}
section[data-testid="stSidebar"] {
    background: #11151d;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 NSE TERMINAL PRO")

st.caption(
    "Advanced Indian Stock Market Research | Technical Analysis | "
    "Fundamentals | Portfolio | Risk | Backtesting"
)

st.info(
    "Educational tool only. Historical data and simulations cannot "
    "guarantee future prices. Not financial advice."
)

# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def ticker(symbol):
    symbol = str(symbol).strip().upper()

    if not symbol:
        return ""

    if symbol.endswith((".NS", ".BO")) or symbol.startswith("^"):
        return symbol

    return symbol + ".NS"


def clean_symbol(symbol):
    return symbol.replace(".NS", "").replace(".BO", "")


def money(value):
    try:
        if pd.notna(value) and np.isfinite(float(value)):
            return f"₹{float(value):,.2f}"
    except:
        pass

    return "N/A"


def safe_float(value):
    try:
        return float(value)
    except:
        return np.nan


# =========================================================
# 3. STOCK DATA
# =========================================================

@st.cache_data(ttl=600, show_spinner=False)
def history(symbol, period="2y"):

    data = yf.Ticker(ticker(symbol)).history(
        period=period,
        interval="1d",
        auto_adjust=True
    )

    if data is None or data.empty:
        return pd.DataFrame()

    data = data.copy()

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    return data.dropna(subset=["Close"])


@st.cache_data(ttl=1800, show_spinner=False)
def fundamentals(symbol):

    result = {}

    try:
        stock = yf.Ticker(ticker(symbol))
        result = stock.get_info()

        if not isinstance(result, dict):
            result = {}

    except:
        pass

    return result


@st.cache_data(ttl=900, show_spinner=False)
def get_news(symbol):

    try:
        return yf.Ticker(ticker(symbol)).news or []

    except:
        return []


# =========================================================
# 4. TECHNICAL INDICATORS
# =========================================================

def rsi(series, period=14):

    change = series.diff()

    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def add_indicators(data):

    df = data.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Moving averages

    for n in [9, 20, 50, 100, 200]:

        df[f"EMA {n}"] = close.ewm(
            span=n,
            adjust=False
        ).mean()

    for n in [20, 50, 100, 200]:

        df[f"SMA {n}"] = close.rolling(n).mean()

    # RSI

    df["RSI 14"] = rsi(close)

    # MACD

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26

    df["MACD Signal"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACD Histogram"] = (
        df["MACD"] - df["MACD Signal"]
    )

    # Bollinger Bands

    middle = close.rolling(20).mean()
    std = close.rolling(20).std()

    df["BB Upper"] = middle + 2 * std
    df["BB Middle"] = middle
    df["BB Lower"] = middle - 2 * std

    # ATR

    previous_close = close.shift()

    tr = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ], axis=1).max(axis=1)

    df["ATR 14"] = tr.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    # ADX

    up = high.diff()
    down = -low.diff()

    plus_dm = up.where(
        (up > down) & (up > 0), 0
    )

    minus_dm = down.where(
        (down > up) & (down > 0), 0
    )

    plus_di = (
        100 * plus_dm.ewm(alpha=1/14).mean()
        / df["ATR 14"]
    )

    minus_di = (
        100 * minus_dm.ewm(alpha=1/14).mean()
        / df["ATR 14"]
    )

    dx = (
        100 * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    df["ADX 14"] = dx.ewm(
        alpha=1/14,
        min_periods=14,
        adjust=False
    ).mean()

    # Momentum indicators

    df["ROC 12"] = close.pct_change(12) * 100

    lowest = low.rolling(14).min()
    highest = high.rolling(14).max()

    df["Stochastic %K"] = (
        100 * (close - lowest)
        / (highest - lowest).replace(0, np.nan)
    )

    df["Stochastic %D"] = (
        df["Stochastic %K"].rolling(3).mean()
    )

    typical_price = (high + low + close) / 3

    tp_mean = typical_price.rolling(20).mean()

    mad = typical_price.rolling(20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True
    )

    df["CCI 20"] = (
        (typical_price - tp_mean)
        / (0.015 * mad.replace(0, np.nan))
    )

    df["Williams %R"] = (
        -100 * (highest - close)
        / (highest - lowest).replace(0, np.nan)
    )

    # Volume indicators

    volume = df["Volume"].fillna(0)

    direction = np.sign(close.diff()).fillna(0)

    df["OBV"] = (direction * volume).cumsum()

    money_multiplier = (
        ((close - low) - (high - close))
        / (high - low).replace(0, np.nan)
    )

    money_volume = money_multiplier.fillna(0) * volume

    positive = money_volume.clip(lower=0).rolling(14).sum()

    negative = (
        -money_volume.clip(upper=0)
    ).rolling(14).sum()

    df["MFI 14"] = (
        100 - 100 / (1 + positive / negative.replace(0, np.nan))
    )

    df["Volume SMA 20"] = volume.rolling(20).mean()

    df["VWAP 20"] = (
        (typical_price * volume).rolling(20).sum()
        / volume.rolling(20).sum().replace(0, np.nan)
    )

    # Donchian channels

    df["Donchian Upper"] = high.rolling(20).max()
    df["Donchian Lower"] = low.rolling(20).min()

    df["Donchian Middle"] = (
        df["Donchian Upper"] + df["Donchian Lower"]
    ) / 2

    # Keltner channels

    df["Keltner Middle"] = close.ewm(
        span=20,
        adjust=False
    ).mean()

    df["Keltner Upper"] = (
        df["Keltner Middle"] + 2 * df["ATR 14"]
    )

    df["Keltner Lower"] = (
        df["Keltner Middle"] - 2 * df["ATR 14"]
    )

    # Ichimoku

    df["Ichimoku Tenkan"] = (
        high.rolling(9).max() + low.rolling(9).min()
    ) / 2

    df["Ichimoku Kijun"] = (
        high.rolling(26).max() + low.rolling(26).min()
    ) / 2

    df["Ichimoku Span A"] = (
        df["Ichimoku Tenkan"] + df["Ichimoku Kijun"]
    ) / 2

    df["Ichimoku Span B"] = (
        high.rolling(52).max() + low.rolling(52).min()
    ) / 2

    # Additional metrics

    df["Daily Return %"] = close.pct_change() * 100
    df["20D Return %"] = close.pct_change(20) * 100

    df["52W High"] = close.rolling(
        252, min_periods=30
    ).max()

    df["52W Low"] = close.rolling(
        252, min_periods=30
    ).min()

    return df


# =========================================================
# 5. SIGNAL GENERATION
# =========================================================

def generate_signal(df):

    last = df.iloc[-1]

    conditions = [
        last["EMA 20"] > last["EMA 50"],
        last["Close"] > last["SMA 50"],
        last["MACD"] > last["MACD Signal"],
        last["RSI 14"] > 50,
        last["ADX 14"] > 20
    ]

    valid = [x for x in conditions if pd.notna(x)]

    score = sum(bool(x) for x in valid)

    if len(valid) == 0:
        return "Insufficient data", score, 0

    if score >= 4:
        return "Bullish", score, len(valid)

    elif score <= 1:
        return "Bearish", score, len(valid)

    return "Neutral", score, len(valid)


# =========================================================
# 6. SESSION STATE
# =========================================================

if "watchlist" not in st.session_state:

    st.session_state.watchlist = [
        "RELIANCE", "TCS", "INFY",
        "HDFCBANK", "ICICIBANK"
    ]

if "alerts" not in st.session_state:

    st.session_state.alerts = []

if "portfolio" not in st.session_state:

    st.session_state.portfolio = pd.DataFrame([
        {"Ticker": "RELIANCE", "Quantity": 1.0, "Buy Price": 100.0},
        {"Ticker": "TCS", "Quantity": 1.0, "Buy Price": 100.0},
        {"Ticker": "INFY", "Quantity": 1.0, "Buy Price": 100.0}
    ])


# =========================================================
# 7. SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Stock Controls")

    symbol = st.text_input(
        "NSE stock ticker",
        "RELIANCE"
    ).strip().upper()

    period = st.selectbox(
        "Historical data",
        ["6mo", "1y", "2y", "5y", "10y"],
        index=2
    )

    chart_days = st.selectbox(
        "Chart window",
        [60, 120, 252, 500, 1000],
        index=2
    )

    st.caption(
        "Examples: RELIANCE, TCS, INFY, "
        "HDFCBANK, SBIN, CIPLA"
    )

    if st.button("🔄 Refresh data"):

        st.cache_data.clear()
        st.rerun()


if not symbol:
    st.stop()


# =========================================================
# 8. LOAD MAIN STOCK
# =========================================================

try:

    raw = history(symbol, period)

except Exception as error:

    st.error(f"Unable to load data: {error}")
    st.stop()


if raw.empty:

    st.error(
        "No stock data found. Please check the NSE ticker."
    )

    st.stop()


if len(raw) < 30:

    st.error("Not enough historical data.")
    st.stop()


df = add_indicators(raw)

close = df["Close"]

price = float(close.iloc[-1])

previous = float(close.iloc[-2])

daily_change = (price / previous - 1) * 100

info = fundamentals(symbol)


# =========================================================
# 9. APPLICATION TABS
# =========================================================

tabs = st.tabs([
    "Overview",
    "Market Overview",
    "Stock Screener",
    "Watchlist & Alerts",
    "Price Chart",
    "Indicators",
    "Growth Scenarios",
    "Risk Analysis",
    "Portfolio",
    "Fundamentals",
    "News",
    "Backtesting",
    "Downloads"
])


# =========================================================
# TAB 1: OVERVIEW
# =========================================================

with tabs[0]:

    st.subheader(f"{symbol} Stock Overview")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Latest price",
        money(price),
        f"{daily_change:+.2f}%"
    )

    c2.metric(
        "RSI 14",
        f"{df['RSI 14'].iloc[-1]:.2f}"
    )

    c3.metric(
        "MACD Histogram",
        f"{df['MACD Histogram'].iloc[-1]:.3f}"
    )

    c4.metric(
        "ADX 14",
        f"{df['ADX 14'].iloc[-1]:.2f}"
    )

    c5.metric(
        "20-day return",
        f"{df['20D Return %'].iloc[-1]:+.2f}%"
    )

    signal, score, total = generate_signal(df)

    if signal == "Bullish":

        st.success(
            f"Technical outlook: {signal} | Score: {score}/{total}"
        )

    elif signal == "Bearish":

        st.error(
            f"Technical outlook: {signal} | Score: {score}/{total}"
        )

    else:

        st.warning(
            f"Technical outlook: {signal} | Score: {score}/{total}"
        )

    view = df.tail(chart_days)

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=view.index,
        open=view["Open"],
        high=view["High"],
        low=view["Low"],
        close=view["Close"],
        name="Price"
    ))

    fig.add_trace(go.Scatter(
        x=view.index,
        y=view["EMA 20"],
        name="EMA 20"
    ))

    fig.add_trace(go.Scatter(
        x=view.index,
        y=view["EMA 50"],
        name="EMA 50"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=500,
        title=f"{symbol} Price Trend",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latest Stock Data")

    st.dataframe(
        df.tail(10).round(2),
        use_container_width=True
    )

    st.caption(
        "Technical scores are rule-based indicators, not guaranteed trade signals."
    )


# =========================================================
# TAB 2: MARKET OVERVIEW
# =========================================================

with tabs[1]:

    st.subheader("Indian Stock Market Overview")

    indices = {
        "NIFTY 50": "^NSEI",
        "BANK NIFTY": "^NSEBANK",
        "SENSEX": "^BSESN"
    }

    cols = st.columns(3)

    for i, (name, code) in enumerate(indices.items()):

        try:

            data = history(code, "1y")

            if len(data) >= 2:

                latest = data["Close"].iloc[-1]
                previous_close = data["Close"].iloc[-2]

                change = (
                    latest / previous_close - 1
                ) * 100

                cols[i].metric(
                    name,
                    f"{latest:,.2f}",
                    f"{change:+.2f}%"
                )

            else:

                cols[i].metric(name, "N/A")

        except:

            cols[i].metric(name, "Unavailable")

    st.divider()

    st.subheader("Stock vs NIFTY 50")

    nifty = history("^NSEI", period)

    if not nifty.empty:

        stock_normalized = (
            close / close.iloc[0]
        ) * 100

        nifty_normalized = (
            nifty["Close"] / nifty["Close"].iloc[0]
        ) * 100

        comparison = pd.concat(
            [
                stock_normalized.rename(symbol),
                nifty_normalized.rename("NIFTY 50")
            ],
            axis=1
        ).dropna()

        fig = px.line(
            comparison,
            template="plotly_dark",
            title="Relative Performance (Starting Value = 100)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning("NIFTY data unavailable.")


# =========================================================
# TAB 3: STOCK SCREENER
# =========================================================

with tabs[2]:

    st.subheader("🔎 Stock Screener")

    st.write(
        "Find stocks in your watchlist using technical indicators."
    )

    min_rsi, max_rsi = st.slider(
        "Select RSI range",
        0, 100, (0, 100)
    )

    filter_type = st.selectbox(
        "Choose stock condition",
        [
            "All Stocks",
            "Bullish Stocks",
            "Bearish Stocks",
            "Price Above SMA 50"
        ]
    )

    if st.button("Run Stock Screener", type="primary"):

        results = []

        progress = st.progress(0)

        symbols = st.session_state.watchlist

        for i, stock in enumerate(symbols):

            try:

                hist = history(stock, "1y")

                if len(hist) < 60:
                    continue

                data = add_indicators(hist)

                last = data.iloc[-1]

                signal, score, total = generate_signal(data)

                rsi_value = safe_float(last["RSI 14"])

                if pd.isna(rsi_value):
                    continue

                if not min_rsi <= rsi_value <= max_rsi:
                    continue

                if filter_type == "Bullish Stocks" and signal != "Bullish":
                    continue

                if filter_type == "Bearish Stocks" and signal != "Bearish":
                    continue

                if filter_type == "Price Above SMA 50":

                    if last["Close"] <= last["SMA 50"]:
                        continue

                results.append({

                    "Stock": clean_symbol(stock),

                    "Price": last["Close"],

                    "RSI": rsi_value,

                    "MACD Histogram": last["MACD Histogram"],

                    "ADX": last["ADX 14"],

                    "20D Return %": last["20D Return %"],

                    "Signal": signal,

                    "Score": f"{score}/{total}"

                })

            except:
                pass

            progress.progress((i + 1) / len(symbols))

        if results:

            result_df = pd.DataFrame(results)

            result_df = result_df.sort_values(
                "20D Return %",
                ascending=False
            )

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Screener Results",
                result_df.to_csv(index=False).encode("utf-8"),
                file_name="stock_screener.csv",
                mime="text/csv"
            )

        else:

            st.warning("No stocks matched your conditions.")

    st.caption(
        "This screener checks your watchlist, not every NSE-listed stock."
    )


# =========================================================
# TAB 4: WATCHLIST AND PRICE ALERTS
# =========================================================

with tabs[3]:

    st.subheader("⭐ My Watchlist")

    col1, col2 = st.columns([2, 1])

    with col1:

        new_stock = st.text_input(
            "Add stock",
            placeholder="Example: SBIN"
        )

    with col2:

        remove_stock = st.selectbox(
            "Remove stock",
            [""] + st.session_state.watchlist
        )

    c1, c2 = st.columns(2)

    with c1:

        if st.button("Add to Watchlist"):

            new_ticker = ticker(new_stock)

            if new_stock and new_ticker not in st.session_state.watchlist:

                st.session_state.watchlist.append(new_ticker)

                st.success("Stock added successfully.")

                st.rerun()

            else:

                st.warning("Stock already exists or is invalid.")

    with c2:

        if st.button("Remove from Watchlist"):

            if remove_stock:

                st.session_state.watchlist.remove(remove_stock)

                st.rerun()

    watch_rows = []

    for stock in st.session_state.watchlist:

        try:

            data = history(stock, "5d")

            if len(data) >= 2:

                latest = data["Close"].iloc[-1]
                previous_close = data["Close"].iloc[-2]

                watch_rows.append({

                    "Stock": clean_symbol(stock),

                    "Latest Price": latest,

                    "Daily Change %": (
                        latest / previous_close - 1
                    ) * 100

                })

        except:
            pass

    if watch_rows:

        st.dataframe(
            pd.DataFrame(watch_rows),
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.subheader("🔔 Price Alerts")

    with st.form("price_alert_form"):

        alert_stock = st.text_input(
            "Stock ticker",
            value=symbol
        )

        target_price = st.number_input(
            "Target price ₹",
            min_value=0.01,
            value=max(price, 1.0)
        )

        condition = st.selectbox(
            "Alert condition",
            [
                "Price reaches above target",
                "Price falls below target"
            ]
        )

        submit_alert = st.form_submit_button(
            "Create Price Alert"
        )

        if submit_alert:

            st.session_state.alerts.append({

                "Stock": ticker(alert_stock),

                "Target": target_price,

                "Condition": condition

            })

            st.success("Price alert created.")

    if st.session_state.alerts:

        alert_results = []

        for alert in st.session_state.alerts:

            try:

                data = history(alert["Stock"], "5d")

                current = float(data["Close"].iloc[-1])

                if alert["Condition"] == "Price reaches above target":

                    triggered = current >= alert["Target"]

                else:

                    triggered = current <= alert["Target"]

                alert_results.append({

                    "Stock": clean_symbol(alert["Stock"]),

                    "Current Price": current,

                    "Target Price": alert["Target"],

                    "Condition": alert["Condition"],

                    "Status": "TRIGGERED" if triggered else "Waiting"

                })

            except:

                continue

        if alert_results:

            st.dataframe(
                pd.DataFrame(alert_results),
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "Alerts are checked when the app reruns. "
            "They are not automatic push notifications."
        )

        if st.button("Clear All Alerts"):

            st.session_state.alerts = []

            st.rerun()


# =========================================================
# TAB 5: PRICE CHART
# =========================================================

with tabs[4]:

    st.subheader("📊 Advanced Price Chart")

    overlays = [
        "EMA 9", "EMA 20", "EMA 50",
        "EMA 100", "EMA 200",
        "SMA 20", "SMA 50", "SMA 100", "SMA 200",
        "BB Upper", "BB Middle", "BB Lower",
        "VWAP 20",
        "Donchian Upper", "Donchian Middle", "Donchian Lower",
        "Keltner Upper", "Keltner Middle", "Keltner Lower",
        "Ichimoku Tenkan", "Ichimoku Kijun",
        "Ichimoku Span A", "Ichimoku Span B"
    ]

    selected = st.multiselect(
        "Select indicator lines",
        overlays,
        default=["EMA 20", "SMA 50"]
    )

    chart_type = st.radio(
        "Chart style",
        ["Candlestick", "Closing Price"],
        horizontal=True
    )

    show_volume = st.checkbox(
        "Show volume",
        value=True
    )

    view = df.tail(chart_days)

    fig = go.Figure()

    if chart_type == "Candlestick":

        fig.add_trace(go.Candlestick(

            x=view.index,

            open=view["Open"],

            high=view["High"],

            low=view["Low"],

            close=view["Close"],

            name="Price"

        ))

    else:

        fig.add_trace(go.Scatter(

            x=view.index,

            y=view["Close"],

            name="Closing Price"

        ))

    for column in selected:

        fig.add_trace(go.Scatter(

            x=view.index,

            y=view[column],

            name=column

        ))

    if show_volume:

        fig.add_trace(go.Bar(

            x=view.index,

            y=view["Volume"],

            name="Volume",

            yaxis="y2",

            opacity=0.25

        ))

        fig.update_layout(

            yaxis2=dict(
                overlaying="y",
                side="right",
                showgrid=False
            )

        )

    fig.update_layout(

        template="plotly_dark",

        height=600,

        title=f"{symbol} Price Analysis",

        xaxis_rangeslider_visible=False,

        hovermode="x unified"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# TAB 6: INDICATORS
# =========================================================

with tabs[5]:

    st.subheader("Technical Indicator Analysis")

    selected_indicator = st.selectbox(

        "Select Indicator",

        [
            "RSI 14",
            "MACD",
            "ADX 14",
            "ATR 14",
            "Daily Return %",
            "Bollinger Bands",
            "Stochastic",
            "ROC 12",
            "CCI 20",
            "Williams %R",
            "MFI 14",
            "OBV",
            "VWAP 20",
            "Donchian Channels",
            "Keltner Channels",
            "Ichimoku",
            "Volume"
        ]

    )

    view = df.tail(chart_days)

    fig = go.Figure()

    if selected_indicator == "MACD":

        for column in [
            "MACD",
            "MACD Signal",
            "MACD Histogram"
        ]:

            fig.add_trace(go.Scatter(

                x=view.index,

                y=view[column],

                name=column

            ))

    elif selected_indicator == "Bollinger Bands":

        for column in [
            "BB Upper",
            "BB Middle",
            "BB Lower",
            "Close"
        ]:

            fig.add_trace(go.Scatter(

                x=view.index,

                y=view[column],

                name=column

            ))

    elif selected_indicator == "Stochastic":

        for column in [
            "Stochastic %K",
            "Stochastic %D"
        ]:

            fig.add_trace(go.Scatter(

                x=view.index,

                y=view[column],

                name=column

            ))

        fig.add_hline(y=80, line_dash="dash")
        fig.add_hline(y=20, line_dash="dash")

    elif selected_indicator in [
        "Donchian Channels",
        "Keltner Channels",
        "Ichimoku"
    ]:

        groups = {

            "Donchian Channels": [
                "Donchian Upper",
                "Donchian Middle",
                "Donchian Lower"
            ],

            "Keltner Channels": [
                "Keltner Upper",
                "Keltner Middle",
                "Keltner Lower"
            ],

            "Ichimoku": [
                "Ichimoku Tenkan",
                "Ichimoku Kijun",
                "Ichimoku Span A",
                "Ichimoku Span B"
            ]

        }

        for column in groups[selected_indicator] + ["Close"]:

            fig.add_trace(go.Scatter(

                x=view.index,

                y=view[column],

                name=column

            ))

    elif selected_indicator == "Volume":

        fig.add_trace(go.Bar(

            x=view.index,

            y=view["Volume"],

            name="Volume"

        ))

        fig.add_trace(go.Scatter(

            x=view.index,

            y=view["Volume SMA 20"],

            name="Volume SMA 20"

        ))

    else:

        fig.add_trace(go.Scatter(

            x=view.index,

            y=view[selected_indicator],

            name=selected_indicator

        ))

        if selected_indicator == "RSI 14":

            fig.add_hline(y=70, line_dash="dash")
            fig.add_hline(y=30, line_dash="dash")

            st.caption(
                "RSI above 70 is often overbought. "
                "RSI below 30 is often oversold."
            )

        elif selected_indicator == "ADX 14":

            fig.add_hline(y=25, line_dash="dash")

            st.caption(
                "ADX measures trend strength, not direction."
            )

        elif selected_indicator == "Williams %R":

            fig.add_hline(y=-20, line_dash="dash")
            fig.add_hline(y=-80, line_dash="dash")

        elif selected_indicator == "MFI 14":

            fig.add_hline(y=80, line_dash="dash")
            fig.add_hline(y=20, line_dash="dash")

    fig.update_layout(

        template="plotly_dark",

        height=450,

        title=selected_indicator

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    indicator_columns = [

        "RSI 14", "MACD", "MACD Signal",
        "MACD Histogram", "ADX 14", "ATR 14",
        "Stochastic %K", "Stochastic %D",
        "ROC 12", "CCI 20", "Williams %R",
        "MFI 14", "OBV", "VWAP 20",
        "Volume SMA 20", "EMA 9", "EMA 20",
        "EMA 50", "EMA 100", "EMA 200",
        "SMA 20", "SMA 50", "SMA 100",
        "SMA 200", "BB Upper", "BB Middle",
        "BB Lower", "Donchian Upper",
        "Donchian Middle", "Donchian Lower",
        "Keltner Upper", "Keltner Middle",
        "Keltner Lower", "Ichimoku Tenkan",
        "Ichimoku Kijun", "Ichimoku Span A",
        "Ichimoku Span B"

    ]

    st.dataframe(

        pd.DataFrame({

            "Indicator": indicator_columns,

            "Latest Value": [
                df[column].iloc[-1]
                for column in indicator_columns
            ]

        }),

        use_container_width=True,

        hide_index=True

    )


# =========================================================
# TAB 7: GROWTH / LOSS SCENARIOS
# =========================================================

with tabs[6]:

    st.subheader("Future Growth and Loss Scenarios")

    horizon = st.select_slider(

        "Forecast horizon (trading days)",

        [7, 30, 90, 252],

        value=30

    )

    simulations = st.select_slider(

        "Number of simulation paths",

        [500, 1000, 2000, 5000],

        value=1000

    )

    log_returns = np.log(
        close / close.shift()
    ).dropna()

    mu = log_returns.mean()
    sigma = log_returns.std()

    rng = np.random.default_rng(42)

    paths = price * np.exp(

        np.cumsum(

            rng.normal(

                mu - 0.5 * sigma**2,

                sigma,

                (simulations, horizon)

            ),

            axis=1

        )

    )

    final_prices = paths[:, -1]

    p5, p25, p50, p75, p95 = np.percentile(

        final_prices,

        [5, 25, 50, 75, 95]

    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Current Price", money(price))

    c2.metric("Median Scenario", money(p50))

    c3.metric("Pessimistic Scenario", money(p5))

    c4.metric("Optimistic Scenario", money(p95))

    scenarios = pd.DataFrame({

        "Scenario": [
            "5th Percentile",
            "25th Percentile",
            "Median",
            "75th Percentile",
            "95th Percentile"
        ],

        "Price": [p5, p25, p50, p75, p95]

    })

    scenarios["Change %"] = (
        scenarios["Price"] / price - 1
    ) * 100

    st.dataframe(
        scenarios,
        use_container_width=True,
        hide_index=True
    )

    q5, q50, q95 = np.percentile(
        paths,
        [5, 50, 95],
        axis=0
    )

    days = np.arange(1, horizon + 1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=days,
        y=q95,
        name="95th Percentile"
    ))

    fig.add_trace(go.Scatter(
        x=days,
        y=q50,
        name="Median"
    ))

    fig.add_trace(go.Scatter(
        x=days,
        y=q5,
        name="5th Percentile",
        fill="tonexty"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=420,
        title="Monte Carlo Price Simulation",
        xaxis_title="Trading Days",
        yaxis_title="Price ₹"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.warning(
        "This is a statistical simulation based on historical returns, "
        "not an AI prediction. Simulation frequencies are not "
        "guaranteed real-world probabilities."
    )


# =========================================================
# TAB 8: RISK ANALYSIS
# =========================================================

with tabs[7]:

    st.subheader("Advanced Risk Analysis")

    returns = close.pct_change().dropna()

    volatility = returns.std() * np.sqrt(252) * 100

    years = len(returns) / 252

    annual_return = (

        (price / close.iloc[0]) ** (1 / years) - 1

    ) * 100 if years > 0 else np.nan

    drawdown = close / close.cummax() - 1

    max_drawdown = drawdown.min() * 100

    daily_var = np.percentile(returns, 5) * 100

    cvar = returns[
        returns <= np.percentile(returns, 5)
    ].mean() * 100

    sharpe = (

        (returns.mean() * 252 - 0.065)
        / (returns.std() * np.sqrt(252))

    ) if returns.std() > 0 else np.nan

    downside = returns[returns < 0]

    downside_deviation = (
        downside.std() * np.sqrt(252)
        if len(downside) > 1 else np.nan
    )

    sortino = (
        (returns.mean() * 252 - 0.065)
        / downside_deviation
        if pd.notna(downside_deviation) and downside_deviation > 0
        else np.nan
    )

    nifty = history("^NSEI", period)

    beta = np.nan
    correlation = np.nan

    if not nifty.empty:

        nifty_returns = nifty["Close"].pct_change()

        aligned = pd.concat(
            [
                returns.rename("Stock"),
                nifty_returns.rename("NIFTY")
            ],
            axis=1
        ).dropna()

        if len(aligned) > 2 and aligned["NIFTY"].var() > 0:

            beta = (
                aligned["Stock"].cov(aligned["NIFTY"])
                / aligned["NIFTY"].var()
            )

            correlation = aligned["Stock"].corr(
                aligned["NIFTY"]
            )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Annualized Volatility",
        f"{volatility:.2f}%"
    )

    c2.metric(
        "Annualized Return",
        f"{annual_return:.2f}%"
    )

    c3.metric(
        "Maximum Drawdown",
        f"{max_drawdown:.2f}%"
    )

    c4.metric(
        "Daily VaR (95%)",
        f"{daily_var:.2f}%"
    )

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(
        "Sharpe Ratio",
        f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A"
    )

    c6.metric(
        "Sortino Ratio",
        f"{sortino:.2f}" if pd.notna(sortino) else "N/A"
    )

    c7.metric(
        "Beta vs NIFTY",
        f"{beta:.2f}" if pd.notna(beta) else "N/A"
    )

    c8.metric(
        "CVaR (95%)",
        f"{cvar:.2f}%"
    )

    st.metric(
        "Correlation with NIFTY 50",
        f"{correlation:.3f}"
        if pd.notna(correlation) else "N/A"
    )

    dd_data = pd.DataFrame({

        "Date": drawdown.index,

        "Drawdown %": drawdown.values * 100

    }).tail(chart_days)

    fig = px.area(

        dd_data,

        x="Date",

        y="Drawdown %",

        template="plotly_dark",

        title="Stock Drawdown Analysis"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Sharpe and Sortino use an assumed 6.5% annual risk-free rate. "
        "Risk measures are historical estimates."
    )


# =========================================================
# TAB 9: PORTFOLIO
# =========================================================

with tabs[8]:

    st.subheader("📁 Portfolio Management")

    st.write(
        "Enter your stock holdings to calculate portfolio performance."
    )

    portfolio = st.data_editor(

        st.session_state.portfolio,

        num_rows="dynamic",

        use_container_width=True,

        key="portfolio_editor"

    )

    st.session_state.portfolio = portfolio

    if st.button("Calculate Portfolio"):

        holdings = []

        for _, row in portfolio.iterrows():

            stock = str(row["Ticker"]).strip().upper()

            qty = safe_float(row["Quantity"])

            buy = safe_float(row["Buy Price"])

            if not stock or qty <= 0 or buy <= 0:
                continue

            try:

                data = history(stock, "5d")

                if data.empty:
                    continue

                current = float(data["Close"].iloc[-1])

                invested = qty * buy

                value = qty * current

                holdings.append({

                    "Stock": clean_symbol(stock),

                    "Quantity": qty,

                    "Buy Price": buy,

                    "Current Price": current,

                    "Invested": invested,

                    "Current Value": value,

                    "Profit/Loss": value - invested,

                    "Return %": (
                        value / invested - 1
                    ) * 100

                })

            except:
                pass

        if holdings:

            pf = pd.DataFrame(holdings)

            invested = pf["Invested"].sum()

            current_value = pf["Current Value"].sum()

            pnl = current_value - invested

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Total Investment", money(invested))

            c2.metric("Current Value", money(current_value))

            c3.metric(
                "Total Profit/Loss",
                money(pnl),
                f"{pnl/invested*100:+.2f}%"
            )

            c4.metric(
                "Total Holdings",
                len(pf)
            )

            st.dataframe(
                pf,
                use_container_width=True,
                hide_index=True
            )

            fig = px.pie(

                pf,

                names="Stock",

                values="Current Value",

                title="Portfolio Allocation",

                template="plotly_dark"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.download_button(

                "Download Portfolio Report",

                pf.to_csv(index=False).encode("utf-8"),

                file_name="portfolio_report.csv",

                mime="text/csv"

            )

        else:

            st.warning(
                "Please enter valid holdings and calculate again."
            )


# =========================================================
# TAB 10: FUNDAMENTALS
# =========================================================

with tabs[9]:

    st.subheader(f"🏢 Fundamental Analysis: {symbol}")

    fields = [

        ("Company", "longName"),

        ("Sector", "sector"),

        ("Industry", "industry"),

        ("Market Capitalization", "marketCap"),

        ("Trailing P/E", "trailingPE"),

        ("Forward P/E", "forwardPE"),

        ("Price to Book", "priceToBook"),

        ("EPS", "trailingEps"),

        ("ROE", "returnOnEquity"),

        ("Profit Margin", "profitMargins"),

        ("Revenue Growth", "revenueGrowth"),

        ("Debt to Equity", "debtToEquity"),

        ("Dividend Yield", "dividendYield"),

        ("52 Week High", "fiftyTwoWeekHigh"),

        ("52 Week Low", "fiftyTwoWeekLow"),

        ("Average Volume", "averageVolume")

    ]

    rows = []

    for label, key in fields:

        value = info.get(key)

        if value is None:
            continue

        if key in [
            "returnOnEquity",
            "profitMargins",
            "revenueGrowth",
            "dividendYield"
        ]:

            value = f"{value * 100:.2f}%"

        elif key == "marketCap":

            value = f"₹{value:,.0f}"

        elif key in [
            "fiftyTwoWeekHigh",
            "fiftyTwoWeekLow",
            "trailingEps"
        ]:

            value = money(value)

        rows.append({

            "Metric": label,

            "Value": value

        })

    if rows:

        st.dataframe(

            pd.DataFrame(rows),

            use_container_width=True,

            hide_index=True

        )

    else:

        st.warning(
            "Fundamental data unavailable for this stock."
        )

    st.caption(
        "Valuation ratios should be compared with sector peers, "
        "company history, profitability and financial strength."
    )


# =========================================================
# TAB 11: NEWS
# =========================================================

with tabs[10]:

    st.subheader(f"📰 Latest News: {symbol}")

    if st.button("Load Latest News"):

        st.cache_data.clear()

        st.rerun()

    news = get_news(symbol)

    if not news:

        st.info("No news available from the provider.")

    else:

        for item in news[:15]:

            content = item.get("content", item)

            headline = content.get(
                "title",
                "Untitled"
            )

            publisher = (

                (content.get("provider") or {}).get("displayName")

                or content.get("publisher")

                or "Unknown publisher"

            )

            url = (

                (content.get("canonicalUrl") or {}).get("url")

                or content.get("link")

            )

            with st.expander(headline):

                st.write(f"Publisher: {publisher}")

                if content.get("pubDate"):

                    st.write(
                        f"Published: {content['pubDate']}"
                    )

                if content.get("summary"):

                    st.write(content["summary"])

                if url:

                    st.link_button(
                        "Read Full Article",
                        url
                    )


# =========================================================
# TAB 12: BACKTESTING
# =========================================================

with tabs[11]:

    st.subheader("🧪 Advanced Strategy Backtesting")

    st.write(
        "Test historical trading strategies using completed-day signals. "
        "Positions are applied from the next trading day."
    )

    if len(df) < 80:

        st.warning(
            "Select a longer historical period for backtesting."
        )

    else:

        strategy = st.selectbox(

            "Choose Trading Strategy",

            [
                "EMA Crossover",
                "RSI Mean Reversion",
                "MACD Crossover",
                "Trend Confirmation"
            ]

        )

        direction = st.selectbox(

            "Trading Mode",

            [
                "Long Only",
                "Long / Short (Simplified)"
            ]

        )

        fee = st.slider(

            "Transaction cost per position change (basis points)",

            0, 50, 10

        )

        capital = st.number_input(

            "Starting Capital ₹",

            min_value=1000,

            max_value=100000000,

            value=100000,

            step=10000

        )

        fast_ema = 20

        slow_ema = 50

        if strategy == "EMA Crossover":

            fast_ema = st.selectbox(

                "Fast EMA",

                [5, 9, 10, 20, 30, 50],

                index=3

            )

            slow_options = [

                n for n in [20, 30, 50, 100, 200]

                if n > fast_ema

            ]

            slow_ema = st.selectbox(

                "Slow EMA",

                slow_options,

                index=0

            )

        bt = df.copy()

        close_bt = bt["Close"]

        # Strategy signals

        if strategy == "EMA Crossover":

            fast_line = close_bt.ewm(
                span=fast_ema,
                adjust=False
            ).mean()

            slow_line = close_bt.ewm(
                span=slow_ema,
                adjust=False
            ).mean()

            signals = (
                fast_line > slow_line
            ).astype(float)

        elif strategy == "RSI Mean Reversion":

            signals = pd.Series(
                0.0,
                index=bt.index
            )

            state = 0.0

            for date, value in bt["RSI 14"].items():

                if pd.notna(value):

                    if value < 30:
                        state = 1.0

                    elif value > 70:
                        state = 0.0

                signals.loc[date] = state

        elif strategy == "MACD Crossover":

            signals = (
                bt["MACD"] > bt["MACD Signal"]
            ).astype(float)

        else:

            signals = (

                (bt["EMA 20"] > bt["EMA 50"])

                & (bt["MACD"] > bt["MACD Signal"])

                & (bt["RSI 14"] > 50)

            ).astype(float)

        if direction == "Long / Short (Simplified)":

            signals = signals * 2 - 1

        # Apply next trading day

        positions = signals.shift(1).fillna(0)

        daily_returns = close_bt.pct_change().fillna(0)

        turnover = positions.diff().abs()

        turnover.iloc[0] = abs(positions.iloc[0])

        transaction_cost = turnover * fee / 10000

        strategy_returns = (
            positions * daily_returns
            - transaction_cost
        )

        equity = capital * (
            1 + strategy_returns
        ).cumprod()

        benchmark = capital * (
            1 + daily_returns
        ).cumprod()

        drawdown = (
            equity / equity.cummax()
        ) - 1

        total_return = (
            equity.iloc[-1] / capital - 1
        )

        years = max(

            (
                bt.index[-1] - bt.index[0]
            ).days / 365.25,

            1 / 365.25

        )

        cagr = (

            (equity.iloc[-1] / capital)
            ** (1 / years) - 1

        ) if equity.iloc[-1] > 0 else -1

        sharpe = (

            strategy_returns.mean()
            / strategy_returns.std()
            * np.sqrt(252)

        ) if strategy_returns.std() > 0 else np.nan

        downside = strategy_returns[
            strategy_returns < 0
        ]

        downside_std = (
            downside.std() * np.sqrt(252)
            if len(downside) > 1 else np.nan
        )

        sortino = (

            strategy_returns.mean() * 252
            / downside_std

        ) if pd.notna(downside_std) and downside_std > 0 else np.nan

        exposure = (
            positions != 0
        ).mean() * 100

        trades = int(
            (turnover > 0).sum()
        )

        # Display metrics

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Strategy Ending Value",
            money(equity.iloc[-1]),
            f"{total_return*100:+.2f}%"
        )

        c2.metric(
            "Buy & Hold Value",
            money(benchmark.iloc[-1]),
            f"{(benchmark.iloc[-1]/capital-1)*100:+.2f}%"
        )

        c3.metric(
            "CAGR",
            f"{cagr*100:.2f}%"
        )

        c4.metric(
            "Maximum Drawdown",
            f"{drawdown.min()*100:.2f}%"
        )

        c5, c6, c7, c8 = st.columns(4)

        c5.metric(
            "Sharpe Ratio",
            f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A"
        )

        c6.metric(
            "Sortino Ratio",
            f"{sortino:.2f}" if pd.notna(sortino) else "N/A"
        )

        c7.metric(
            "Position Changes",
            trades
        )

        c8.metric(
            "Market Exposure",
            f"{exposure:.2f}%"
        )

        # Equity chart

        fig = go.Figure()

        fig.add_trace(go.Scatter(

            x=equity.index,

            y=equity,

            name="Strategy"

        ))

        fig.add_trace(go.Scatter(

            x=benchmark.index,

            y=benchmark,

            name="Buy and Hold"

        ))

        fig.update_layout(

            template="plotly_dark",

            height=450,

            title="Strategy vs Buy-and-Hold Performance",

            yaxis_title="Portfolio Value ₹"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Drawdown chart

        ddfig = go.Figure()

        ddfig.add_trace(go.Scatter(

            x=drawdown.index,

            y=drawdown * 100,

            name="Drawdown",

            fill="tozeroy"

        ))

        ddfig.update_layout(

            template="plotly_dark",

            height=300,

            title="Strategy Drawdown",

            yaxis_title="Drawdown %"

        )

        st.plotly_chart(
            ddfig,
            use_container_width=True
        )

        # EMA parameter comparison

        st.subheader("EMA Strategy Comparison")

        st.caption(
            "Compare historical EMA combinations. "
            "The best past combination is not necessarily the best future strategy."
        )

        if st.button("Compare EMA Combinations"):

            combinations = [
                (5, 20),
                (9, 20),
                (9, 50),
                (20, 50),
                (20, 100),
                (50, 200)
            ]

            results = []

            for fast, slow in combinations:

                fast_line = close_bt.ewm(
                    span=fast,
                    adjust=False
                ).mean()

                slow_line = close_bt.ewm(
                    span=slow,
                    adjust=False
                ).mean()

                pos = (
                    fast_line > slow_line
                ).astype(float).shift(1).fillna(0)

                turn = pos.diff().abs()

                turn.iloc[0] = abs(pos.iloc[0])

                returns = (
                    pos * daily_returns
                    - turn * fee / 10000
                )

                curve = (
                    1 + returns
                ).cumprod()

                dd = (
                    curve / curve.cummax()
                ) - 1

                results.append({

                    "EMA Combination": f"{fast}/{slow}",

                    "Total Return %": (
                        curve.iloc[-1] - 1
                    ) * 100,

                    "Maximum Drawdown %": (
                        dd.min()
                    ) * 100,

                    "Position Changes": int(
                        (turn > 0).sum()
                    )

                })

            result_df = pd.DataFrame(results)

            result_df = result_df.sort_values(
                "Total Return %",
                ascending=False
            )

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )

        # Detailed backtest data

        detail = pd.DataFrame({

            "Close": close_bt,

            "Signal": signals,

            "Position": positions,

            "Daily Return %": daily_returns * 100,

            "Strategy Return %": strategy_returns * 100,

            "Transaction Cost %": transaction_cost * 100,

            "Strategy Equity": equity,

            "Benchmark Equity": benchmark,

            "Drawdown %": drawdown * 100

        })

        with st.expander("View Detailed Backtest Results"):

            st.dataframe(
                detail.tail(100).round(3),
                use_container_width=True
            )

        st.download_button(

            "Download Backtest Results",

            detail.to_csv().encode("utf-8"),

            file_name="backtest_results.csv",

            mime="text/csv"

        )

        st.warning(
            "Historical backtesting does not fully model slippage, taxes, "
            "corporate actions, liquidity, brokerage, margin or short-selling "
            "constraints. Results are simulations, not predictions."
        )


# =========================================================
# TAB 13: DOWNLOAD CENTRE
# =========================================================

with tabs[12]:

    st.subheader("⬇️ Download Centre")

    st.write(
        "Download historical prices and calculated technical indicators."
    )

    export_period = st.selectbox(

        "Select download period",

        [
            "Full Selected Period",
            "Last 1 Year",
            "Last 6 Months"
        ]

    )

    if export_period == "Last 1 Year":

        export_data = df.tail(252)

    elif export_period == "Last 6 Months":

        export_data = df.tail(126)

    else:

        export_data = df

    st.download_button(

        "Download Complete Stock Analysis CSV",

        export_data.to_csv().encode("utf-8"),

        file_name=f"{symbol}_complete_analysis.csv",

        mime="text/csv",

        use_container_width=True

    )

    st.download_button(

        "Download Historical Prices CSV",

        raw.to_csv().encode("utf-8"),

        file_name=f"{symbol}_historical_prices.csv",

        mime="text/csv",

        use_container_width=True

    )

    st.markdown("### Export Information")

    st.write(f"**Stock:** {symbol}")

    st.write(f"**Records:** {len(export_data)}")

    st.write(f"**Starting date:** {export_data.index.min().date()}")

    st.write(f"**Last available date:** {export_data.index.max().date()}")

    st.caption(
        "CSV files can be opened in Microsoft Excel."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    f"NSE Terminal Pro | Selected stock: {symbol} | "
    f"Latest available data: {df.index[-1].date()} | "
    "Data source: Yahoo Finance through yfinance"
)
