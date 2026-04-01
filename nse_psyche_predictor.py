import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time
import yfinance as yf
import pandas_ta_classic as ta
import feedparser
from textblob import TextBlob

st.set_page_config(page_title="NSE F&O PsychePredictor", layout="wide")
st.title("🧠 NSE F&O PsychePredictor – Optimized Fast Refresh")
st.markdown("**8-Agent System with News + Psychology** → Simple UP / DOWN command")

# Session state
if 'weights' not in st.session_state:
    st.session_state.weights = {"tech": 35, "psych": 25, "pcr": 15, "emotion": 10, "risk": 30, "news": 20}

# ==================== MARKET DATA (NSE Option Chain) ====================
@st.cache_data(ttl=3)
def fetch_nse_data(symbol: str):
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", timeout=5)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        resp = session.get(url, headers=headers, timeout=8)
        data = resp.json()
        
        records = data['records']
        underlying = records['underlyingValue']
        expiries = sorted([d['expiryDate'] for d in records['data'] if 'expiryDate' in d])
        expiry = expiries[0] if expiries else None
        
        ce_oi = pe_oi = 0
        for item in records['data']:
            if item.get('expiryDate') == expiry:
                ce_oi += item.get('CE', {}).get('openInterest', 0)
                pe_oi += item.get('PE', {}).get('openInterest', 0)
        
        pcr = round(pe_oi / ce_oi, 3) if ce_oi > 0 else 0.9
        
        return {
            "symbol": symbol,
            "underlying_price": round(underlying, 2),
            "pcr": pcr,
            "timestamp": datetime.now().strftime("%H:%M:%S IST")
        }
    except Exception as e:
        st.warning(f"NSE fetch issue: {str(e)[:60]}... Using fallback price.")
        return {"symbol": symbol, "underlying_price": 22679.4, "pcr": 0.83, "timestamp": "Fallback"}

# ==================== TECHNICAL AGENT (Fixed yfinance tickers) ====================
def technical_agent(symbol: str):
    # Correct index tickers for yfinance
    if symbol == "NIFTY":
        ticker = "^NSEI"
    elif symbol == "BANKNIFTY":
        ticker = "^NSEBANK"
    else:
        ticker = f"{symbol}.NS"
    
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False, threads=False, timeout=10)
        if len(df) < 14:
            raise ValueError("Not enough data")
        
        df['RSI'] = ta.rsi(df['Close'], length=14)
        latest = df.iloc[-1]
        rsi = round(latest.get('RSI', 50), 2)
        ma20 = df['Close'].rolling(20).mean().iloc[-1]
        trend = "Strong Up" if latest['Close'] > ma20 and rsi > 55 else \
                "Strong Down" if latest['Close'] < ma20 and rsi < 45 else "Sideways"
        
        return {"rsi": rsi, "trend": trend}
    except Exception as e:
        # Fallback when yfinance fails (common for indices sometimes)
        st.info(f"yfinance issue for {ticker}: using neutral technicals")
        return {"rsi": 50.0, "trend": "Sideways"}

# ==================== Other Agents (same as optimized version) ====================
def psych_agent(data, tech):
    pcr = data.get('pcr', 0.9)
    if pcr < 0.85:
        emotion = "🔥 GREED"
        bias = "Bullish bias"
    elif pcr > 1.25:
        emotion = "😨 FEAR"
        bias = "Bearish bias"
    else:
        emotion = "😐 NEUTRAL"
        bias = "Balanced"
    impulse = "HIGH" if tech['rsi'] > 75 or tech['rsi'] < 25 else "Low"
    return {"emotion": emotion, "bias": bias, "impulse": impulse, "pcr": pcr}

def risk_agent(psych, tech):
    risk_score = 40
    if psych['impulse'] == "HIGH":
        risk_score += 35
    if "GREED" in psych['emotion'] or "FEAR" in psych['emotion']:
        risk_score += 20
    return {"risk_score": min(risk_score, 100)}

@st.cache_data(ttl=180)
def news_agent(symbol: str):
    feeds = ["https://www.moneycontrol.com/rss/latestnews.xml", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"]
    headlines = []
    sentiment_score = 0.0
    count = 0
    keywords = [symbol.lower(), "nifty", "banknifty", "oil", "iran", "stt", "vix", "fii"]
    
    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:12]:
                title = entry.title.lower()
                if any(kw in title for kw in keywords):
                    headlines.append(entry.title)
                    sentiment_score += TextBlob(entry.title).sentiment.polarity
                    count += 1
        except:
            continue
    avg_sent = round(sentiment_score / max(count, 1), 2)
    bias = "Bullish" if avg_sent > 0.08 else "Bearish" if avg_sent < -0.08 else "Neutral"
    verdict = f"🟢 Positive" if bias == "Bullish" else f"🔴 Negative" if bias == "Bearish" else "⚪ Neutral"
    return {"headlines": headlines[:5], "sentiment": avg_sent, "verdict": verdict, "bias": bias}

def prediction_agent(data, tech, psych, risk, news, weights):
    score = 0.0
    if tech.get('trend') in ["Strong Up", "Bullish"]: score += weights["tech"]
    elif tech.get('trend') in ["Strong Down", "Bearish"]: score -= weights["tech"]
    
    if psych.get('bias') == "Bullish bias": score += weights["psych"]
    elif psych.get('bias') == "Bearish bias": score -= weights["psych"]
    
    pcr = data.get('pcr', 0.9)
    if pcr < 0.9: score += weights["pcr"]
    elif pcr > 1.3: score -= weights["pcr"]
    
    if "GREED" in psych.get('emotion', ""): score += weights["emotion"]
    elif "FEAR" in psych.get('emotion', ""): score -= weights["emotion"]
    
    if risk.get('risk_score', 0) > 60: score -= weights["risk"]
    if news["bias"] == "Bullish": score += weights["news"]
    elif news["bias"] == "Bearish": score -= weights["news"]
    
    if score >= 55:
        pred = "🚀 BULLISH"
        conf = min(85, 50 + int(score / 2))
        strategy = "Buy Call / Bull Spread (small)"
    elif score <= -45:
        pred = "📉 BEARISH"
        conf = min(85, 50 + int(abs(score) / 2))
        strategy = "Buy Put / Bear Spread or cash"
    else:
        pred = "⚖️ NEUTRAL"
        conf = 55
        strategy = "Avoid directional"
    
    if risk.get('risk_score', 0) > 70:
        pred += " – HIGH RISK"
        conf = max(35, conf - 20)
    
    return {"prediction": pred, "confidence": conf, "strategy": strategy}

# ==================== FAST REFRESH FRAGMENT ====================
@st.fragment(run_every="15s")
def live_refresh_fragment(symbol, weights):
    with st.spinner("🔄 Live refresh..."):
        data = fetch_nse_data(symbol)
        tech = technical_agent(symbol)
        psych = psych_agent(data, tech)
        risk = risk_agent(psych, tech)
        news = news_agent(symbol)
        pred = prediction_agent(data, tech, psych, risk, news, weights)
        
        st.session_state.data = data
        st.session_state.tech = tech
        st.session_state.psych = psych
        st.session_state.risk = risk
        st.session_state.news = news
        st.session_state.pred = pred

# ==================== MAIN UI ====================
symbol = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY", "FINNIFTY"], key="symbol_select")

live_refresh_fragment(symbol, st.session_state.weights)

st.caption(f"Last refreshed: {st.session_state.get('data', {}).get('timestamp', '—')}")

# Big Bold Prediction
pred = st.session_state.get('pred', {})
direction = "UP" if "BULLISH" in pred.get("prediction", "") else "DOWN" if "BEARISH" in pred.get("prediction", "") else "SIDEWAYS"
color = "🟢" if "BULLISH" in pred.get("prediction", "") else "🔴" if "BEARISH" in pred.get("prediction", "") else "⚪"

st.markdown(f"""
<div style="text-align:center; padding:40px; background:#1e1e1e; border-radius:20px; margin:20px 0;">
    <h1 style="font-size:6rem; margin:0;">{color} {direction}</h1>
    <h2>{pred.get('prediction', 'NEUTRAL')}</h2>
    <h3>Confidence: <span style="color:#00ff88;">{pred.get('confidence', 55)}%</span></h3>
    <p><strong>Strategy:</strong> {pred.get('strategy', '')}</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📡 Data", "📊 Tech", "🧠 Psych", "🛡️ Risk", "📰 News", "⚙️ Weights"])

with tab1:
    d = st.session_state.get('data', {})
    st.metric("Price", f"₹{d.get('underlying_price', 22679.4)}")
    st.metric("PCR", d.get('pcr', 0.83))

with tab2:
    t = st.session_state.get('tech', {})
    st.metric("RSI", t.get('rsi', 50))
    st.metric("Trend", t.get('trend', "Sideways"))

with tab3:
    p = st.session_state.get('psych', {})
    st.markdown(p.get('emotion', "NEUTRAL"))
    st.write("Bias:", p.get('bias', ""))

with tab4:
    r = st.session_state.get('risk', {})
    st.progress(r.get('risk_score', 50)/100)
    st.write("Risk Score:", r.get('risk_score', 50))

with tab5:
    n = st.session_state.get('news', {})
    st.markdown(n.get('verdict', "Neutral"))
    st.metric("Sentiment", n.get('sentiment', 0))
    if n.get('headlines'):
        for h in n['headlines']:
            st.caption(h)

with tab6:
    st.subheader("Agent Weights")
    cols = st.columns(3)
    with cols[0]:
        st.session_state.weights["tech"] = st.slider("Technical", 20, 50, st.session_state.weights["tech"])
        st.session_state.weights["psych"] = st.slider("Psych", 15, 40, st.session_state.weights["psych"])
    with cols[1]:
        st.session_state.weights["pcr"] = st.slider("PCR", 5, 25, st.session_state.weights["pcr"])
        st.session_state.weights["emotion"] = st.slider("Emotion", 5, 20, st.session_state.weights["emotion"])
    with cols[2]:
        st.session_state.weights["risk"] = st.slider("Risk Penalty", 15, 40, st.session_state.weights["risk"])
        st.session_state.weights["news"] = st.slider("News Weight", 10, 30, st.session_state.weights["news"])
    if st.button("Apply Weights"):
        st.success("Weights updated!")

st.caption("Fixed yfinance ticker issue (^NSEI / ^NSEBANK). Optimized with fragments. Educational tool only – not trading advice.")
