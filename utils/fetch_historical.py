"""
Fetch historical data from Binance US API to warm up indicators
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

# Ensure project root (JustHodl-Strategy) is on sys.path so 'utils' is importable when run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import (
    BINANCE_KLINES_PATH,
    BINANCE_PROXY_KLINES_PATH,
    BINANCE_PROXY_URL,
    BINANCE_URL,
    USE_BINANCE_PROXY,
)


def _resolve_klines_endpoint() -> Tuple[str, str, bool]:
    """Determine which base URL/path to use for klines requests."""
    if USE_BINANCE_PROXY and BINANCE_PROXY_URL:
        return BINANCE_PROXY_URL.rstrip("/"), BINANCE_PROXY_KLINES_PATH, True
    return BINANCE_URL.rstrip("/"), BINANCE_KLINES_PATH, False


def _parse_proxy_payload(payload: Dict) -> Optional[List[Dict]]:
    """Normalize proxy response into list of klines."""
    if not payload.get("success", False):
        error_msg = payload.get("error", "Unknown error")
        print(f"Proxy error: {error_msg}")
        return None
    data = payload.get("data", [])
    if not isinstance(data, list) or not data:
        return []
    return data


def _normalize_klines(raw: List) -> pd.DataFrame:
    """Convert proxy or direct kline payloads into OHLCV dataframe."""
    if not raw:
        return pd.DataFrame()

    if isinstance(raw[0], dict):
        df = pd.DataFrame(raw)
        df["timestamp"] = pd.to_datetime(df["openTime"], unit="ms", utc=True)
        df = df.rename(
            columns={
                "openPrice": "open",
                "highPrice": "high",
                "lowPrice": "low",
                "closePrice": "close",
            }
        )
    else:
        cols = [
            "openTime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "closeTime",
            "quoteAssetVolume",
            "numberOfTrades",
            "takerBuyBaseVolume",
            "takerBuyQuoteVolume",
            "ignore",
        ]
        df = pd.DataFrame(raw, columns=cols)
        df["timestamp"] = pd.to_datetime(df["openTime"], unit="ms", utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df.set_index("timestamp", inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


def fetch_binance_klines(symbol: str, interval: str = "5m", limit: int = 500) -> pd.DataFrame:
    """
    Fetch recent candles from Binance (direct) or proxy API.

    Args:
        symbol: Binance symbol (e.g., "BTCUSDT")
        interval: Timeframe (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
        limit: Number of candles to fetch (max 1000)

    Returns:
        DataFrame with OHLCV data
    """

    base_url, path, using_proxy = _resolve_klines_endpoint()
    url = f"{base_url}{path}"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000),
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        payload = response.json()

        if using_proxy:
            normalized = _parse_proxy_payload(payload)
            if normalized is None:
                return pd.DataFrame()
        else:
            normalized = payload

        df = _normalize_klines(normalized)
        if df.empty:
            print(f"No data returned for {symbol} from {'proxy' if using_proxy else 'Binance'} API")
        return df

    except requests.exceptions.RequestException as exc:
        print(f"Network error fetching {symbol}: {exc}")
        return pd.DataFrame()
    except Exception as exc:
        print(f"Error parsing klines for {symbol}: {exc}")
        import traceback

        traceback.print_exc()
        return pd.DataFrame()

def convert_to_binance_symbol(pair: str) -> str:
    """
    Convert trading pair to Binance US symbol format.
    
    Args:
        pair: Trading pair like "BTC/USD" or "BTC/USDT"
    
    Returns:
        Binance symbol like "BTCUSDT"
    """
    # Handle both /USD and /USDT formats
    if "/USDT" in pair:
        return pair.replace("/USDT", "USDT")
    elif "/USD" in pair:
        return pair.replace("/USD", "USDT")
    else:
        # Fallback: just remove the slash
        return pair.replace("/", "")


def fetch_historical_for_symbols(pairs: List[str], interval: str = "5m", limit: int = 500) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple symbols from Binance US.
    
    Args:
        pairs: List of trading pairs (e.g., ["BTC/USD", "ETH/USD"])
        interval: Timeframe (1m, 5m, 15m, 30m, 1h, etc.)
        limit: Number of candles per symbol (max 1000)
    
    Returns:
        Dict mapping trading pairs to DataFrames
    """
    historical_data = {}
    
    for pair in pairs:
        binance_symbol = convert_to_binance_symbol(pair)
        print(f"Fetching {limit} {interval} candles for {pair} ({binance_symbol})...")
        df = fetch_binance_klines(binance_symbol, interval, limit)
        
        if not df.empty:
            historical_data[pair] = df
            print(f"  ✓ Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        else:
            print(f"  ✗ Failed to fetch data for {pair}")
        
        # Rate limiting
        time.sleep(0.2)
    
    return historical_data


def fetch_binance_ticker(symbols: Optional[List[str]] = None) -> Dict[str, Dict]:
    """
    Fetch rolling window price change statistics from Binance API.
    Returns a dict keyed by symbol with the 24hr ticker fields.
    """
    base = f"{BINANCE_URL.rstrip('/')}/api/v3/ticker/24hr"
    result: Dict[str, Dict] = {}
    try:
        if symbols is None:
            r = requests.get(base, timeout=15)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                for t in data:
                    if 'symbol' in t:
                        result[t['symbol']] = t
        elif len(symbols) == 1:
            r = requests.get(base, params={'symbol': symbols[0]}, timeout=15)
            r.raise_for_status()
            t = r.json()
            if 'symbol' in t:
                result[t['symbol']] = t
        else:
            # Safest approach: request per symbol to avoid API incompatibility with symbols=[] param
            for sym in symbols:
                try:
                    r = requests.get(base, params={'symbol': sym}, timeout=15)
                    r.raise_for_status()
                    t = r.json()
                    if 'symbol' in t:
                        result[t['symbol']] = t
                except Exception:
                    continue
        return result
    except Exception:
        return result

if __name__ == "__main__":
    # Test fetching from Binance US
    print("Testing Binance US API...")
    print("=" * 60)
    
    test_pairs = ["BTC/USD", "ETH/USD", "BNB/USD"]
    
    # Test klines
    binance_data = fetch_historical_for_symbols(test_pairs, interval="5m", limit=300)
    
    print("\nBinance US Klines Summary:")
    for pair, df in binance_data.items():
        if not df.empty:
            print(f"{pair}: {len(df)} candles, latest close: ${df['close'].iloc[-1]:.2f}")
    
    # Test ticker endpoint
    print("\n" + "=" * 60)
    print("Testing Binance US Ticker API...")
    print("=" * 60)
    
    # Fetch ticker for specific symbols
    ticker_symbols = ["BTCUSDT", "BNBUSDT"]
    tickers = fetch_binance_ticker(ticker_symbols)
    
    print(f"\nFetched ticker data for {len(tickers)} symbols:")
    for symbol, ticker in tickers.items():
        print(f"\n{symbol}:")
        print(f"  Last Price: ${float(ticker['lastPrice']):.2f}")
        print(f"  Price Change: ${float(ticker['priceChange']):.2f} ({ticker['priceChangePercent']}%)")
        print(f"  24h High: ${float(ticker['highPrice']):.2f}")
        print(f"  24h Low: ${float(ticker['lowPrice']):.2f}")
        print(f"  24h Volume: {float(ticker['volume']):.2f}")
        print(f"  Weighted Avg Price: ${float(ticker['weightedAvgPrice']):.2f}")
    
    print("\n" + "=" * 60)
