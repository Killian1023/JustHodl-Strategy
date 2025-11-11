"""
Technical Indicators for Feature Engineering

Implements common technical indicators used in trading:
- Momentum: RSI, MACD, Rate of Change
- Trend: EMA, SMA, ADX
- Volatility: Bollinger Bands, ATR
- Volume: Volume SMA, OBV
- Price patterns: Returns, high-low ratios
"""

import numpy as np
import pandas as pd
from typing import List


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Price array
        period: RSI period (default 14)
        
    Returns:
        RSI values (0-100)
    """
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    
    if down == 0:
        return np.full_like(prices, 100.0)
    
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)
    
    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        
        if down == 0:
            rsi[i] = 100.0
        else:
            rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)
    
    return rsi


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        prices: Price array
        period: EMA period
        
    Returns:
        EMA values
    """
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    
    multiplier = 2.0 / (period + 1)
    
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema


def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Simple Moving Average (SMA)."""
    sma = np.convolve(prices, np.ones(period)/period, mode='same')
    # Fix edges
    sma[:period-1] = np.cumsum(prices[:period-1]) / np.arange(1, period)
    return sma


def calculate_macd(prices: np.ndarray, 
                   fast_period: int = 12,
                   slow_period: int = 26,
                   signal_period: int = 9) -> tuple:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(prices: np.ndarray, 
                               period: int = 20,
                               num_std: float = 2.0) -> tuple:
    """
    Calculate Bollinger Bands.
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = calculate_sma(prices, period)
    
    # Calculate rolling standard deviation
    std = np.zeros_like(prices)
    for i in range(period-1, len(prices)):
        std[i] = np.std(prices[i-period+1:i+1])
    
    # For early values, use expanding window
    for i in range(period-1):
        std[i] = np.std(prices[:i+1])
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    return upper, middle, lower


def calculate_atr(high: np.ndarray, 
                  low: np.ndarray, 
                  close: np.ndarray,
                  period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range (ATR).
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period
        
    Returns:
        ATR values
    """
    # Calculate True Range
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    
    # First value doesn't have previous close
    high_close[0] = 0
    low_close[0] = 0
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    
    # Calculate ATR using exponential moving average
    atr = np.zeros_like(close)
    atr[0] = true_range[0]
    
    multiplier = 1.0 / period
    for i in range(1, len(close)):
        atr[i] = ((period - 1) * atr[i-1] + true_range[i]) * multiplier
    
    return atr


def calculate_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    Calculate On-Balance Volume (OBV).
    
    Args:
        close: Close prices
        volume: Volume
        
    Returns:
        OBV values
    """
    obv = np.zeros_like(close)
    obv[0] = volume[0]
    
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    
    return obv


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to dataframe.
    
    Args:
        df: DataFrame with OHLCV columns
        
    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
    # === Momentum Indicators ===
    
    # RSI at different periods
    df['rsi_14'] = calculate_rsi(close, 14)
    df['rsi_28'] = calculate_rsi(close, 28)
    
    # MACD
    macd, macd_signal, macd_hist = calculate_macd(close)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist
    
    # Rate of Change (ROC)
    df['roc_12'] = ((close - np.roll(close, 12)) / np.roll(close, 12) * 100)
    df['roc_24'] = ((close - np.roll(close, 24)) / np.roll(close, 24) * 100)
    
    # === Trend Indicators ===
    
    # Moving averages
    df['ema_9'] = calculate_ema(close, 9)
    df['ema_21'] = calculate_ema(close, 21)
    df['ema_50'] = calculate_ema(close, 50)
    df['sma_20'] = calculate_sma(close, 20)
    df['sma_50'] = calculate_sma(close, 50)
    
    # Distance from moving averages
    df['close_ema9_ratio'] = close / df['ema_9'].values
    df['close_sma20_ratio'] = close / df['sma_20'].values
    
    # === Volatility Indicators ===
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    df['bb_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-9)
    df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)
    
    # ATR
    df['atr_14'] = calculate_atr(high, low, close, 14)
    df['atr_normalized'] = df['atr_14'] / close  # Normalized by price
    
    # === Volume Indicators ===
    
    df['volume_sma_20'] = calculate_sma(volume, 20)
    df['volume_ratio'] = volume / (df['volume_sma_20'].values + 1e-10)
    
    # OBV
    df['obv'] = calculate_obv(close, volume)
    df['obv_ema'] = calculate_ema(df['obv'].values, 20)
    
    # === Price Pattern Features ===
    
    # Returns at different horizons
    df['return_1'] = np.log(close / np.roll(close, 1))
    df['return_2'] = np.log(close / np.roll(close, 2))
    df['return_4'] = np.log(close / np.roll(close, 4))
    df['return_8'] = np.log(close / np.roll(close, 8))
    
    # High-Low range
    df['high_low_ratio'] = (high - low) / (close + 1e-10)
    df['high_close_ratio'] = (high - close) / (close + 1e-10)
    df['low_close_ratio'] = (close - low) / (close + 1e-10)
    
    # Price position in range
    df['close_position'] = (close - low) / (high - low + 1e-10)
    
    # === Volatility Measures ===
    
    # Rolling standard deviation of returns
    for period in [5, 10, 20]:
        rolling_returns = pd.Series(df['return_1'].values).rolling(period).std()
        df[f'volatility_{period}'] = rolling_returns.fillna(0).values
    
    # Fill NaN values from indicators that need history
    df = df.bfill().fillna(0)
    
    return df


def get_indicator_feature_names() -> List[str]:
    """
    Get list of all technical indicator feature names.
    
    Returns:
        List of feature column names
    """
    return [
        # Momentum
        'rsi_14', 'rsi_28',
        'macd', 'macd_signal', 'macd_hist',
        'roc_12', 'roc_24',
        
        # Trend
        'ema_9', 'ema_21', 'ema_50',
        'sma_20', 'sma_50',
        'close_ema9_ratio', 'close_sma20_ratio',
        
        # Volatility
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
        'atr_14', 'atr_normalized',
        
        # Volume
        'volume_sma_20', 'volume_ratio',
        'obv', 'obv_ema',
        
        # Price patterns
        'return_1', 'return_2', 'return_4', 'return_8',
        'high_low_ratio', 'high_close_ratio', 'low_close_ratio',
        'close_position',
        
        # Volatility measures
        'volatility_5', 'volatility_10', 'volatility_20',
    ]


def get_selected_features() -> List[str]:
    """
    Get a curated list of most important features.
    
    Returns:
        List of selected feature names
    """
    return [
        # Original OHLCV
        'open', 'high', 'low', 'close', 'volume',
        
        # Key momentum
        'rsi_14', 'macd', 'macd_hist',
        
        # Key trend
        'ema_21', 'sma_20', 'close_sma20_ratio',
        
        # Key volatility
        'bb_width', 'bb_position', 'atr_normalized',
        
        # Key volume
        'volume_ratio',
        
        # Key price patterns
        'return_1', 'return_4',
        'high_low_ratio',
        
        # Volatility
        'volatility_10',
    ]
