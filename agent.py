from flask import Flask, request, jsonify, render_template_string
import re
import os
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import yfinance as yf
from pycoingecko import CoinGeckoAPI
import requests  

app = Flask(__name__)

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')  
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

cg = CoinGeckoAPI()
blacklist_upper = {'A', 'AN', 'AND', 'THE', 'OF', 'FOR', 'TO', 'IN', 'ON', 'AT', 'BY', 'WITH', 'FROM', 'UP', 'OUT', 'PRICE', 'STOCK', 'SHARE', 'BUY', 'SELL', 'HIGH', 'LOW', 'OPEN', 'CLOSE'}
blacklist_lower = {w.lower() for w in blacklist_upper}

known_crypto_ids = {
    "bitcoin", "ethereum", "tether", "xrp", "binancecoin", "solana", "usd-coin", "lido-staked-ether", "tron", "dogecoin",
    "cardano", "wrapped-steth", "wrapped-bitcoin", "whitebit-token", "chainlink", "bitcoin-cash", "zcash", "stellar",
    "hedera-hashgraph", "litecoin", "sui", "avalanche-2", "monero", "shiba-inu", "polkadot", "toncoin", "dai",
    "uniswap", "internet-computer", "bittensor", "near-protocol"
}

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

def crypto_tool(query: str) -> str:
    q = query.lower()
    symbol_map = {
        "btc": "bitcoin", "eth": "ethereum", "usdt": "tether",
        "bnb": "binancecoin", "xrp": "ripple", "ada": "cardano",
        "sol": "solana", "doge": "dogecoin", "matic": "matic-network"
    }
    coin_id = None
    for sym, cid in symbol_map.items():
        if re.search(r'\b' + re.escape(sym) + r'\b', q):
            coin_id = cid
            break
    if not coin_id:
        potentials = [w for w in re.findall(r"\b([a-z]{2,25})\b", q) if w not in blacklist_lower]
        if not potentials:
            return "Please specify a known crypto (e.g., BTC, ETH, Bitcoin)."
        coin_id = potentials[-1]
        if coin_id not in known_crypto_ids:
            return "Please specify a known crypto (e.g., BTC, ETH, Bitcoin)."
    try:
        data = cg.get_price(ids=coin_id, vs_currencies='usd',
                            include_24hr_change=True, include_market_cap=True, include_24hr_vol=True)
        if not data or coin_id not in data:
            coin_list = cg.get_coins_list()
            matched = [c for c in coin_list if coin_id == c.get("id") or coin_id == (c.get("symbol") or "").lower() or coin_id == (c.get("name") or "").lower()]
            if matched:
                new_coin_id = matched[0]["id"]
                if new_coin_id not in known_crypto_ids:
                    return f"Could not find major crypto data for '{query}'."
                coin_id = new_coin_id
                data = cg.get_price(ids=coin_id, vs_currencies='usd',
                                    include_24hr_change=True, include_market_cap=True, include_24hr_vol=True)
        if not data or coin_id not in data:
            return f"Could not find crypto data for '{query}'."
        d = data[coin_id]
        price = safe_float(d.get('usd'))
        change24 = safe_float(d.get('usd_24h_change'))
        marketcap = safe_float(d.get('usd_market_cap'))
        vol24 = safe_float(d.get('usd_24hr_vol'))
        price_str = f"${price:,.4f}" if price < 1 else f"${price:,.2f}"
        mc_str = f"${marketcap:,.0f}" if marketcap else "N/A"
        vol_str = f"${vol24:,.0f}" if vol24 else "N/A"
        # Get rank
        rank_str = "N/A"
        try:
            markets = cg.get_coins_markets(vs_currency='usd', ids=coin_id, per_page=1, order='market_cap_desc')
            if markets:
                rank_str = f"#{markets[0]['market_cap_rank']}"
        except:
            pass
        wisdom = f"\n\nWisdom: {coin_id.title()} embodies innovation but volatility. HODL wisely, research deeply, and remember: in crypto, fortune favors the informed bold."
        return (f"Crypto {coin_id.replace('-', ' ').title()} — Price: {price_str}, "
                f"24h Change: {change24:+.2f}%, Rank: {rank_str}, MCap: {mc_str}, 24h Vol: {vol_str}{wisdom}")
    except Exception as e:
        return f"Error fetching crypto data: {str(e)}"

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

# LangGraph State
class AgentState(TypedDict):
    messages: list
    category: str
    result: str

# Nodes
def classify_query(state: AgentState) -> AgentState:
    """Classify query into stocks, crypto, or forex using LLM"""
    if not llm:
        # Fallback to regex classification
        q_lower = state["messages"][-1].content.lower()
        if any(word in q_lower for word in ["stock", "share", "ticker"]) or re.search(r'\b[A-Z]{1,4}\b', state["messages"][-1].content):
            category = "stocks"
        elif any(word in q_lower for word in ["crypto", "bitcoin", "eth", "coin"]):
            category = "crypto"
        elif any(word in q_lower for word in ["forex", "currency", "usd", "eur", "convert"]):
            category = "forex"
        else:
            category = "stocks"  # Default
    else:
        prompt = ChatPromptTemplate.from_template(
            "Classify the query into one of: stocks, crypto, forex. Respond with only the category.\nQuery: {query}"
        )
        chain = prompt | llm | StrOutputParser()
        category = chain.invoke({"query": state["messages"][-1].content}).strip().lower()
        if category not in ["stocks", "crypto", "forex"]:
            category = "stocks"
    return {"messages": state["messages"], "category": category}

def route_to_tool(state: AgentState) -> AgentState:
    """Route to the appropriate tool based on category"""
    query = state["messages"][-1].content
    category = state["category"]
    if category == "stocks":
        result = stock_tool(query)
    elif category == "crypto":
        result = crypto_tool(query)
    elif category == "forex":
        result = forex_tool(query)
    else:
        result = "Could not classify query. Try specifying stocks, crypto, or forex."
    return {"messages": state["messages"] + [AIMessage(content=result)], "category": category, "result": result}

# Build Graph
workflow = StateGraph(state_schema=AgentState)
workflow.add_node("classify", classify_query)
workflow.add_node("route_tool", route_to_tool)
workflow.set_entry_point("classify")
workflow.add_edge("classify", "route_tool")
workflow.add_edge("route_tool", END)

app_graph = workflow.compile()

def process_query(query: str) -> str:
    """Process query using LangGraph"""
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "category": None,
        "result": None
    }
    final_state = app_graph.invoke(initial_state)
    return final_state["result"]

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Investment Assistant — Auto-Detect Mode</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-primary: #0a0a0f;
      --bg-secondary: #1a1a24;
      --bg-card: #212130;
      --text-primary: #ffffff;
      --text-secondary: #b0b0b8;
      --accent-primary: #00d4ff;
      --accent-secondary: #6366f1;
      --border-color: #333344;
      --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
      --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
      --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
      --border-radius: 12px;
      --transition: all 0.2s ease-in-out;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
      color: var(--text-primary);
      padding: 2rem 1rem;
      min-height: 100vh;
      line-height: 1.6;
    }

    .container {
      max-width: 800px;
      margin: 0 auto;
    }

    .header {
      text-align: center;
      margin-bottom: 2rem;
    }

    .header h1 {
      font-size: 2.5rem;
      font-weight: 700;
      background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin-bottom: 0.5rem;
    }

    .header p {
      color: var(--text-secondary);
      font-size: 1.1rem;
      font-weight: 400;
    }

    .card {
      background: var(--bg-card);
      border-radius: var(--border-radius);
      padding: 2rem;
      box-shadow: var(--shadow-lg);
      border: 1px solid var(--border-color);
      transition: var(--transition);
    }

    .card:hover {
      transform: translateY(-2px);
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }

    .form-group {
      margin-bottom: 1.5rem;
    }

    label {
      display: block;
      margin-bottom: 0.5rem;
      font-weight: 500;
      color: var(--text-primary);
      font-size: 0.95rem;
    }

    input[type="text"] {
      width: 100%;
      padding: 0.875rem 1rem;
      font-size: 1rem;
      border: 1px solid var(--border-color);
      border-radius: var(--border-radius);
      background: var(--bg-secondary);
      color: var(--text-primary);
      transition: var(--transition);
      font-family: inherit;
    }

    input[type="text"]:focus {
      outline: none;
      border-color: var(--accent-primary);
      box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1);
    }

    .examples {
      margin-bottom: 1.5rem;
      padding: 1rem;
      background: rgba(0, 212, 255, 0.05);
      border-radius: var(--border-radius);
      border-left: 3px solid var(--accent-primary);
    }

    .examples code {
      background: var(--bg-secondary);
      padding: 0.25rem 0.5rem;
      border-radius: 6px;
      font-family: 'Monaco', monospace;
      color: var(--accent-primary);
    }

    .button-group {
      display: flex;
      gap: 0.75rem;
      margin-top: 1rem;
    }

    button {
      flex: 1;
      padding: 0.875rem 1.5rem;
      font-size: 1rem;
      font-weight: 500;
      border: none;
      border-radius: var(--border-radius);
      cursor: pointer;
      transition: var(--transition);
      font-family: inherit;
    }

    #ask {
      background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
      color: white;
    }

    #ask:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3);
    }

    #clear {
      background: var(--bg-secondary);
      color: var(--text-secondary);
      border: 1px solid var(--border-color);
    }

    #clear:hover {
      background: var(--border-color);
      color: var(--text-primary);
    }

    h3 {
      margin: 1.5rem 0 1rem;
      font-size: 1.25rem;
      font-weight: 600;
      color: var(--text-primary);
    }

    #out {
      white-space: pre-wrap;
      background: var(--bg-secondary);
      padding: 1.5rem;
      border-radius: var(--border-radius);
      border: 1px solid var(--border-color);
      font-family: 'Inter', monospace;
      font-size: 0.95rem;
      line-height: 1.6;
      color: var(--text-primary);
      min-height: 120px;
      overflow-y: auto;
    }

    #out::before {
      content: '⟨ ';
      color: var(--accent-primary);
    }

    #out::after {
      content: ' ⟩';
      color: var(--accent-primary);
    }

    @media (max-width: 768px) {
      body {
        padding: 1rem 0.5rem;
      }

      .header h1 {
        font-size: 2rem;
      }

      .card {
        padding: 1.5rem;
      }

      .button-group {
        flex-direction: column;
      }
    }

    /* Loading animation */
    .loading {
      display: inline-block;
      width: 20px;
      height: 20px;
      border: 3px solid var(--bg-secondary);
      border-radius: 50%;
      border-top-color: var(--accent-primary);
      animation: spin 1s ease-in-out infinite;
      margin-right: 0.5rem;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Investment Assistant</h1>
      <p>Auto-Detect Intelligence for Stocks | Crypto | Forex</p>
    </div>
    <div class="card">
      <p style="color: var(--text-secondary); margin-bottom: 1.5rem; text-align: center;">
        LangGraph-powered auto-detection delivers precise insights. Enhanced by wisdom and real-time data from top APIs.
      </p>
      
      <div class="examples">
        <strong>Examples:</strong> <code>AAPL price</code>, <code>Apple</code>, <code>price of ETH</code>, <code>Bitcoin</code>, <code>Convert 100 USD to INR</code>
      </div>
      
      <div class="form-group">
        <input id="q" type="text" placeholder="Ask about stocks, crypto, or forex... (auto-detected)" />
      </div>
      
      <div class="button-group">
        <button id="ask">Ask <span id="loading" class="loading" style="display:none;"></span></button>
        <button id="clear">Clear</button>
      </div>
      
      <h3>Response</h3>
      <pre id="out">(Awaiting your query...)</pre>
    </div>
  </div>

<script>
const qInput = document.getElementById('q');
const out = document.getElementById('out');
const ask = document.getElementById('ask');
const clearBtn = document.getElementById('clear');
const loading = document.getElementById('loading');

async function send(query) {
  out.textContent = '⟨ Thinking... ⟩';
  loading.style.display = 'inline-block';
  ask.disabled = true;
  try {
    const resp = await fetch('/query', {
      method: 'POST', 
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({q: query})
    });
    const data = await resp.json();
    out.textContent = data.result;
  } catch (error) {
    out.textContent = '⟨ An error occurred. Please try again. ⟩';
  } finally {
    loading.style.display = 'none';
    ask.disabled = false;
  }
}

ask.addEventListener('click', () => {
  const q = qInput.value.trim();
  if (q) send(q);
});

qInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    const q = qInput.value.trim();
    if (q) send(q);
  }
});

clearBtn.addEventListener('click', () => {
  qInput.value = '';
  out.textContent = '(Awaiting your query...)';
});
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/query', methods=['POST'])
def query_route():
    data = request.get_json() or {}
    q = (data.get('q') or '').strip()
    if not q:
        return jsonify({'result': 'Please send a query.'})
    try:
        res = process_query(q)
        return jsonify({'result': res})
    except Exception as e:
        return jsonify({'result': f'Agent error: {str(e)}'})

if __name__ == '__main__':
    print('Starting Investment Assistant — LangGraph Auto-Detect Mode — open http://127.0.0.1:5000')
    print('Set POLYGON_API_KEY and OPENAI_API_KEY env vars for optimal performance.')
    app.run(debug=True)