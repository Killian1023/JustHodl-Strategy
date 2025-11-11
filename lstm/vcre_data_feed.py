"""
VCRE-style Data Feed for LSTM Strategy
Provides real-time 15-minute candle data using Binance US REST API polling
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional, Deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fetch_historical import fetch_historical_for_symbols, fetch_binance_klines, convert_to_binance_symbol

logger = logging.getLogger(__name__)


@dataclass
class Candle:
    """OHLCV Candle"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class CandleBuffer:
    """Manages candle data for a symbol"""
    
    def __init__(self, symbol: str, timeframe_minutes: int, max_candles: int = 500):
        self.symbol = symbol
        self.timeframe_minutes = timeframe_minutes
        self.max_candles = max_candles
        self.candles: Deque[Candle] = deque(maxlen=max_candles)
        self.last_candle_time: Optional[datetime] = None
    
    def load_historical_data(self, df: pd.DataFrame):
        """Load historical candles from DataFrame"""
        for timestamp, row in df.iterrows():
            candle = Candle(
                timestamp=timestamp.to_pydatetime().replace(tzinfo=None),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume'])
            )
            self.candles.append(candle)
        
        if len(self.candles) > 0:
            self.last_candle_time = self.candles[-1].timestamp
            logger.debug(f"[{self.symbol}] Historical data loaded up to {self.last_candle_time}")
    
    def add_candle(self, candle: Candle) -> bool:
        """
        Add a new complete candle to the buffer.
        Returns True if this is a new candle (not a duplicate).
        """
        # If we already have a candle for this timestamp, replace it (partial candle update)
        if self.last_candle_time is not None:
            if candle.timestamp < self.last_candle_time:
                # Older candle, ignore
                return False
            if candle.timestamp == self.last_candle_time:
                # Same timestamp: update the last candle in-place (still-forming candle)
                if len(self.candles) > 0:
                    self.candles.pop()
                self.candles.append(candle)
                # last_candle_time unchanged
                logger.debug(f"[{self.symbol}] Updated partial candle: {candle.timestamp} | Close: ${candle.close:.2f}")
                return False
        
        # Add the candle
        self.candles.append(candle)
        self.last_candle_time = candle.timestamp
        
        logger.debug(f"[{self.symbol}] New candle added: {candle.timestamp} | Close: ${candle.close:.2f}")
        return True
    
    def get_dataframe(self) -> pd.DataFrame:
        """Convert candles to DataFrame"""
        if len(self.candles) == 0:
            return pd.DataFrame()
        
        data = {
            'timestamp': [c.timestamp for c in self.candles],
            'open': [c.open for c in self.candles],
            'high': [c.high for c in self.candles],
            'low': [c.low for c in self.candles],
            'close': [c.close for c in self.candles],
            'volume': [c.volume for c in self.candles]
        }
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def has_enough_data(self, min_candles: int) -> bool:
        """Check if we have enough candles"""
        return len(self.candles) >= min_candles


class VCREDataFeed:
    """
    Data feed adapter using VCRE-style Binance US REST API polling.
    Provides 15-minute candle data for LSTM strategy.
    """
    
    # Binance interval mapping
    INTERVAL_MAP = {
        1: "1m",
        5: "5m",
        15: "15m",
        30: "30m",
        60: "1h"
    }
    
    def __init__(self, symbol: str, timeframe_minutes: int = 15, max_candles: int = 500):
        """
        Initialize data feed.
        
        Args:
            symbol: Trading pair in Roostoo format (e.g., "BTC/USD")
            timeframe_minutes: Candle timeframe in minutes (default: 15)
            max_candles: Maximum number of candles to keep in buffer
        """
        self.symbol = symbol
        self.timeframe_minutes = timeframe_minutes
        self.max_candles = max_candles
        
        # Initialize candle buffer
        self.candle_buffer = CandleBuffer(symbol, timeframe_minutes, max_candles)
        
        # Track last poll time
        self.last_poll_time: Optional[datetime] = None
        
        logger.info(f"VCREDataFeed initialized for {symbol} ({timeframe_minutes}m)")
    
    def _get_interval_string(self) -> str:
        """Get Binance interval string for configured timeframe"""
        return self.INTERVAL_MAP.get(self.timeframe_minutes, "15m")
    
    def load_historical_warmup(self, num_candles: int = 300):
        """
        Load historical data to warm up the buffer.
        
        Args:
            num_candles: Number of historical candles to fetch (max 1000)
        """
        logger.info("=" * 60)
        logger.info("Loading Historical Data for Warmup (Binance US)")
        logger.info("=" * 60)
        logger.info(f"Fetching {num_candles} {self.timeframe_minutes}-minute candles for {self.symbol}...")
        
        # Binance US max is 1000 candles
        num_candles = min(num_candles, 1000)
        
        # Fetch historical data
        interval = self._get_interval_string()
        historical_data = fetch_historical_for_symbols([self.symbol], interval=interval, limit=num_candles)
        
        # Load into candle buffer
        if self.symbol in historical_data and not historical_data[self.symbol].empty:
            df = historical_data[self.symbol]
            self.candle_buffer.load_historical_data(df)
            last_time = df.index[-1]
            last_price = df['close'].iloc[-1]
            logger.info(f"  ✓ {self.symbol}: {len(df)} candles loaded")
            logger.info(f"    Latest: {last_time} | Price: ${last_price:.2f}")
        else:
            logger.warning(f"  ✗ {self.symbol}: No historical data available")
        
        logger.info("=" * 60)
        logger.info(f"✅ Historical warmup complete: {len(self.candle_buffer.candles)} candles")
        logger.info("=" * 60)
    
    def update_latest_candles(self, num_candles: int = 10):
        """
        Fetch and update latest candles from Binance.
        
        Args:
            num_candles: Number of recent candles to fetch (default: 10)
        """
        try:
            # Convert symbol to Binance format
            binance_symbol = convert_to_binance_symbol(self.symbol)
            interval = self._get_interval_string()
            
            # Fetch latest candles
            df = fetch_binance_klines(binance_symbol, interval=interval, limit=num_candles)
            
            if df.empty:
                logger.warning(f"[{self.symbol}] No new candles fetched")
                return
            
            # Add new candles to buffer
            new_count = 0
            for timestamp, row in df.iterrows():
                candle = Candle(
                    timestamp=timestamp.to_pydatetime().replace(tzinfo=None),
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume'])
                )
                if self.candle_buffer.add_candle(candle):
                    new_count += 1
            
            if new_count > 0:
                logger.info(f"[{self.symbol}] Added {new_count} new candles")
            
            self.last_poll_time = datetime.now()
            
        except Exception as e:
            logger.error(f"[{self.symbol}] Error updating candles: {e}")
    
    def get_latest_data(self, lookback_periods: int = 100, include_partial: bool = True) -> pd.DataFrame:
        """
        Get latest N candles as DataFrame.
        
        Args:
            lookback_periods: Number of recent candles to return
            include_partial: Whether to include the still-forming (unclosed) latest candle
            
        Returns:
            DataFrame with columns: timestamp (index), open, high, low, close, volume
        """
        df = self.candle_buffer.get_dataframe()
        
        if df.empty:
            logger.warning(f"[{self.symbol}] No data available in buffer")
            return pd.DataFrame()
        
        # Optionally drop the still-forming last candle (not yet closed)
        if not include_partial and len(df) > 0 and self.candle_buffer.last_candle_time is not None:
            last_ts = df.index[-1]
            # Compare in UTC (df index is UTC-aware from fetch)
            now_utc = datetime.utcnow()
            if isinstance(last_ts, pd.Timestamp) and last_ts.tzinfo is not None:
                # Convert to naive UTC for comparison consistency
                last_ts_naive = last_ts.tz_convert(None).to_pydatetime()
            else:
                last_ts_naive = last_ts.to_pydatetime() if isinstance(last_ts, pd.Timestamp) else last_ts
            candle_close_time = last_ts_naive + timedelta(minutes=self.timeframe_minutes)
            if now_utc < candle_close_time:
                # Drop the last (partial) candle
                if len(df) > 1:
                    df = df.iloc[:-1]
                else:
                    # If only one candle and it's partial, return empty
                    return pd.DataFrame()
        
        # Return last N periods
        result = df.tail(lookback_periods).reset_index()
        return result
    
    def get_current_price(self) -> Optional[float]:
        """Get current price (close of latest candle)"""
        if len(self.candle_buffer.candles) == 0:
            return None
        return self.candle_buffer.candles[-1].close
    
    def has_enough_data(self, min_candles: int = 60) -> bool:
        """Check if buffer has enough data for strategy"""
        return self.candle_buffer.has_enough_data(min_candles)
    
    def get_data_info(self) -> Dict:
        """Get information about current data state"""
        return {
            'symbol': self.symbol,
            'timeframe_minutes': self.timeframe_minutes,
            'total_candles': len(self.candle_buffer.candles),
            'last_candle_time': self.candle_buffer.last_candle_time,
            'last_poll_time': self.last_poll_time,
            'current_price': self.get_current_price()
        }
