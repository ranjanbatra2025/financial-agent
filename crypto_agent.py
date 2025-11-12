# crypto_agent.py
import re
from typing import Any
from pycoingecko import CoinGeckoAPI

cg = CoinGeckoAPI()

blacklist_lower = {'a', 'an', 'and', 'the', 'of', 'for', 'to', 'in', 'on', 'at', 'by', 'with', 'from', 'up', 'out', 'price', 'stock', 'share', 'buy', 'sell', 'high', 'low', 'open', 'close'}

known_crypto_ids = {
    "bitcoin", "ethereum", "tether", "xrp", "binancecoin", "solana", "usd-coin", "lido-staked-ether", "tron", "dogecoin",
    "cardano", "wrapped-steth", "wrapped-bitcoin", "whitebit-token", "chainlink", "bitcoin-cash", "zcash", "stellar",
    "hedera-hashgraph", "litecoin", "sui", "avalanche-2", "monero", "shiba-inu", "polkadot", "toncoin", "dai",
    "uniswap", "internet-computer", "bittensor", "near-protocol"
}

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

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