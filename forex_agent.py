# forex_agent.py
import re
import os
from typing import Any, Tuple
import requests
import yfinance as yf

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def polygon_forex_data(pair: str) -> float:
    """Fetch forex rate from Polygon.io"""
    if not POLYGON_API_KEY:
        raise ValueError("POLYGON_API_KEY not set")
    symbol = f"{pair.replace('/', '')}=X"  # e.g., EURUSD=X
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?apikey={POLYGON_API_KEY}"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise ValueError(f"Polygon API error: {resp.text}")
    data = resp.json()
    if 'results' not in data or not data['results']:
        raise ValueError("No data from Polygon")
    return data['results'][-1]['c']

def forex_tool(query: str) -> str:
    q = query.upper()
    pair = None
    amount = 1.0
    m = re.search(r'([A-Z]{3})\s*[/\-]\s*([A-Z]{3})', q)
    if m:
        pair = (m.group(1), m.group(2))
    else:
        m2 = re.search(r'convert\s+([\d,\.]+)\s*([A-Z]{3})\s+to\s+([A-Z]{3})', q, re.I)
        if m2:
            amount = safe_float(m2.group(1).replace(',', ''))
            pair = (m2.group(2).upper(), m2.group(3).upper())
        else:
            m3 = re.search(r'([A-Z]{3})\s+to\s+([A-Z]{3})', q, re.I)
            if m3:
                pair = (m3.group(1).upper(), m3.group(2).upper())
    if not pair:
        return "Please specify a currency pair like 'USD/INR' or 'Convert 100 USD to EUR'. Tip: Forex rates fluctuate based on economic indicators like interest rates and inflation."
    base, quote = pair
    pair_str = f"{base}/{quote}"
    try:
        rate = polygon_forex_data(pair_str)
        converted = amount * rate
        insight = " Economic Insight: Monitor central bank policies for volatility."
        wisdom = f"\n\nWisdom: Forex like {base}/{quote} thrives on global events. Trade with discipline—use stop-losses, leverage wisely, and let economics, not emotions, guide you."
        if amount != 1.0:
            return f"Forex {base}/{quote} — {amount} {base} = {converted:,.4f} {quote} (rate: 1 {base} = {rate:.6f} {quote}){insight}{wisdom}"
        else:
            return f"Forex {base}/{quote} — Rate: 1 {base} = {rate:.6f} {quote}.{insight}{wisdom}"
    except Exception as e:
        # Fallback to yf
        symbol = f"{base}{quote}=X"
        try:
            tk = yf.Ticker(symbol)
            info = tk.info or {}
            rate = safe_float(info.get("regularMarketPrice") or info.get("previousClose", 0))
            if rate == 0.0:
                return f"Could not fetch rate for {base}/{quote}."
            converted = amount * rate
            insight = " Economic Insight: Monitor central bank policies for volatility."
            wisdom = f"\n\nWisdom: Forex like {base}/{quote} thrives on global events. Trade with discipline—use stop-losses, leverage wisely, and let economics, not emotions, guide you."
            if amount != 1.0:
                return f"Forex {base}/{quote} — {amount} {base} = {converted:,.4f} {quote} (rate: 1 {base} = {rate:.6f} {quote}){insight}{wisdom}"
            else:
                return f"Forex {base}/{quote} — Rate: 1 {base} = {rate:.6f} {quote}.{insight}{wisdom}"
        except Exception as e2:
            return f"Error fetching forex rate: {str(e2)}"