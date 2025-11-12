# stock_agent.py
import re
import os
from typing import Dict, Any
import yfinance as yf
import requests

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

stock_name_map = {
    "apple": "AAPL", "microsoft": "MSFT", "alphabet": "GOOGL", "google": "GOOGL",
    "amazon": "AMZN", "meta": "META", "facebook": "META", "tesla": "TSLA",
    "nvidia": "NVDA", "netflix": "NFLX", "adobe": "ADBE", "broadcom": "AVGO",
    "oracle": "ORCL", "salesforce": "CRM", "ibm": "IBM", "intel": "INTC",
    "amd": "AMD", "cisco": "CSCO", "walmart": "WMT", "visa": "V",
    "mastercard": "MA", "jpmorgan": "JPM", "berkshire": "BRK-B", "johnson": "JNJ",
    "procter": "PG", "unitedhealth": "UNH", "home depot": "HD", "chevron": "CVX",
    "exxon": "XOM", "coca cola": "KO", "pepsi": "PEP"
}

blacklist_upper = {'A', 'AN', 'AND', 'THE', 'OF', 'FOR', 'TO', 'IN', 'ON', 'AT', 'BY', 'WITH', 'FROM', 'UP', 'OUT', 'PRICE', 'STOCK', 'SHARE', 'BUY', 'SELL', 'HIGH', 'LOW', 'OPEN', 'CLOSE'}

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def polygon_stock_data(ticker: str) -> Dict[str, Any]:
    """Fetch stock data from Polygon.io"""
    if not POLYGON_API_KEY:
        raise ValueError("POLYGON_API_KEY not set")
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?adjusted=true&apikey={POLYGON_API_KEY}"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise ValueError(f"Polygon API error: {resp.text}")
    data = resp.json()
    if 'results' not in data or not data['results']:
        raise ValueError("No data from Polygon")
    latest = data['results'][-1]
    prev = data['results'][-2] if len(data['results']) > 1 else latest
    return {
        'latest_close': latest['c'],
        'prev_close': prev['c'],
        'volume': latest['v'],
        'high_52': 0,  
        'low_52': 0,
        'name': ticker 
    }

def stock_tool(query: str) -> str:
    q_lower = query.lower()
    ticker = None
    for name in stock_name_map:
        if re.search(r'\b' + re.escape(name) + r'\b', q_lower):
            ticker = stock_name_map[name]
            break
    if ticker is None:
        potentials = [t for t in re.findall(r"\b([A-Z]{1,4})\b", query.upper()) if t not in blacklist_upper and len(t) >= 2]
        if potentials:
            ticker = potentials[0]
    if not ticker:
        return "Please specify a stock ticker or company name (e.g., AAPL, Apple, MSFT)."
    try:
        data = polygon_stock_data(ticker)
        latest = safe_float(data['latest_close'])
        prev = safe_float(data['prev_close'])
        pct = ((latest - prev) / prev * 100) if prev and prev != 0 else 0.0
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        name = info.get("shortName") or info.get("longName") or ticker
        market_cap = safe_float(info.get('marketCap', 0))
        high_52 = safe_float(info.get('fiftyTwoWeekHigh', 0))
        low_52 = safe_float(info.get('fiftyTwoWeekLow', 0))
        volume = safe_float(data['volume'])  
        extra = f" | MCap: ${market_cap:,.0f} | Vol: {volume:,.0f} | 52W: ${low_52:.2f}-${high_52:.2f}"
        news_str = ""
        try:
            news = tk.news
            if news and len(news) > 0:
                latest_news = news[0]
                title = latest_news.get('title', '')
                publisher = latest_news.get('publisher', 'Unknown')
                if isinstance(publisher, dict):
                    publisher = publisher.get('name', 'Unknown')
                news_str = f"\nLatest News: {title} ({publisher})"
        except:
            news_str = ""
        wisdom = f"\n\nWisdom: Investing in stocks like {name} rewards patience. Focus on fundamentals, diversify, and avoid emotional trades—markets reward the disciplined."
        return f"Stock {ticker} ({name}) — Price: ${latest:.2f}, Change: {pct:+.2f}%{extra}{news_str}{wisdom}"
    except Exception as e:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="3d")
            info = tk.info or {}
            if hist is None or hist.empty:
                latest = safe_float(info.get("regularMarketPrice") or info.get("previousClose"))
                if latest == 0.0:
                    return f"No recent data found for {ticker}."
                prev = safe_float(info.get("previousClose", latest))
            else:
                closes = list(hist["Close"])
                latest = safe_float(closes[-1])
                prev = safe_float(closes[-2]) if len(closes) > 1 else safe_float(info.get("previousClose", latest))
            pct = ((latest - prev) / prev * 100) if prev and prev != 0 else 0.0
            name = info.get("shortName") or info.get("longName") or ticker
            market_cap = safe_float(info.get('marketCap', 0))
            volume = safe_float(info.get('volume', 0))
            high_52 = safe_float(info.get('fiftyTwoWeekHigh', 0))
            low_52 = safe_float(info.get('fiftyTwoWeekLow', 0))
            extra = f" | MCap: ${market_cap:,.0f} | Vol: {volume:,.0f} | 52W: ${low_52:.2f}-${high_52:.2f}"
            news_str = ""
            try:
                news = tk.news
                if news and len(news) > 0:
                    latest_news = news[0]
                    title = latest_news.get('title', '')
                    publisher = latest_news.get('publisher', 'Unknown')
                    if isinstance(publisher, dict):
                        publisher = publisher.get('name', 'Unknown')
                    news_str = f"\nLatest News: {title} ({publisher})"
            except:
                news_str = ""
            wisdom = f"\n\nWisdom: Investing in stocks like {name} rewards patience. Focus on fundamentals, diversify, and avoid emotional trades—markets reward the disciplined."
            return f"Stock {ticker} ({name}) — Price: ${latest:.2f}, Change: {pct:+.2f}%{extra}{news_str}{wisdom}"
        except Exception as e2:
            return f"Error fetching stock data for {ticker}: {str(e2)}"