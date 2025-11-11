"""
Live Trading Bot using Alma Strategy (ALMA + RR TP/SL) with Roostoo API
- Single or multiple coins with per-symbol optimized params (optional)
- Poll 1-second data every 5 seconds, aggregate to 15m base and alt timeframe = base*alt_multiplier
- Entry via market order. Stop loss via market (on breach). Take profit via pre-placed limit order.
"""

import sys
import os
import time
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

# Add the package root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roostoo_trading import RoostooClient, RoostooAPIError, RoostooOrderError
from roostoo_trading.config import ROOSTOO_CONFIG
from utils.fetch_historical import fetch_binance_klines, convert_to_binance_symbol
from utils.supabase_logger import SupabaseLogger
# Reuse core functions from backtest_alma_pine
from backtest_alma_pine import Params as AlmaParams, resample_ohlcv, variant
# Import Alma-specific config
from live_alma_config import (
    QUANTITY_STEP_SIZES,
    PRICE_STEP_SIZES,
    PORTFOLIO_CONFIG,
    LOG_CONFIG,
    RATE_LIMIT_CONFIG,
)


logging.basicConfig(
    level=getattr(logging, LOG_CONFIG.get('log_level', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_CONFIG.get('log_file', 'live_Alma_trader.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    entry_time: datetime
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    order_id: Optional[int] = None
    tp_order_id: Optional[int] = None
    trade_id: Optional[str] = None


class TickAggregator:
    """Aggregate 1-second ticks to configurable minute timeframe for multiple timeframes"""
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ticks: List[dict] = []  # each: {ts, open, high, low, close, volume}

    def add_tick(self, price: float, volume: float, ts: Optional[datetime] = None):
        ts = ts or datetime.utcnow()
        self.ticks.append({
            'timestamp': ts.replace(microsecond=0),
            'open': price, 'high': price, 'low': price, 'close': price, 'volume': volume or 0.0
        })
        # keep last 7 days of 1s ticks
        cutoff = datetime.utcnow() - timedelta(days=7)
        self.ticks = [t for t in self.ticks if t['timestamp'] >= cutoff]

    def to_df(self) -> pd.DataFrame:
        if not self.ticks:
            return pd.DataFrame(columns=['open','high','low','close','volume'])
        df = pd.DataFrame(self.ticks)
        df.set_index('timestamp', inplace=True)
        # ensure sorted
        df = df.sort_index()
        return df

    def resample_minutes(self, minutes: int) -> pd.DataFrame:
        df = self.to_df()
        if df.empty:
            return df
        return resample_ohlcv(df, minutes)


class AlmaLiveTrader:
    def __init__(self, symbols: List[str], base_tf_min: int = 15, poll_seconds: int = 5,
                 params_dir: Optional[str] = None,
                 enable_supabase: bool = True, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        self.symbols = symbols
        self.base_tf_min = base_tf_min
        self.poll_seconds = poll_seconds

        self.roostoo = RoostooClient(api_key=ROOSTOO_CONFIG['api_key'], secret_key=ROOSTOO_CONFIG['secret_key'])
        self.supabase_logger = SupabaseLogger(supabase_url=supabase_url, supabase_key=supabase_key, enabled=enable_supabase)

        self.params_dir = params_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'optimization_results_Alma')
        self.aggregators: Dict[str, TickAggregator] = {s: TickAggregator(s) for s in symbols}
        self.positions: Dict[str, Position] = {}
        self.equity = PORTFOLIO_CONFIG.get('initial_capital', 10000.0)
        self.cash = self.equity

        # Load per-symbol params if available
        self.params_map: Dict[str, AlmaParams] = {}
        for s in symbols:
            self.params_map[s] = self._load_params_for_symbol(s)

        # Warmup: fetch recent base timeframe candles and seed base history per symbol
        self.base_history: Dict[str, pd.DataFrame] = {s: pd.DataFrame(columns=['open','high','low','close','volume']) for s in symbols}
        try:
            self._warmup_base_history(limit=200)
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

        logger.info(f"AlmaLiveTrader initialized, symbols={symbols}")

    def _load_params_for_symbol(self, symbol: str) -> AlmaParams:
        # Expect file: optimization_results_Alma/<SYM>/<SYM>_best.json
        try:
            sym = symbol.replace('/', '')
            path = os.path.join(self.params_dir, sym, f"{sym}_best.json")
            if os.path.exists(path):
                with open(path, 'r') as f:
                    j = json.load(f)
                bp = j.get('best_params', {})
                p = AlmaParams(csv_path='', base_tf_min=self.base_tf_min)
                p.alt_multiplier = int(bp.get('alt_multiplier', p.alt_multiplier))
                p.basis_len = int(bp.get('basis_len', p.basis_len))
                p.offset_sigma = int(bp.get('offset_sigma', p.offset_sigma))
                p.offset_alma = float(bp.get('offset_alma', p.offset_alma))
                p.last_n_len = int(bp.get('last_n_len', p.last_n_len))
                p.rr_multiplier = float(bp.get('rr_multiplier', p.rr_multiplier))
                return p
        except Exception as e:
            logger.warning(f"[{symbol}] could not load optimized params: {e}")
        # defaults
        return AlmaParams(csv_path='', base_tf_min=self.base_tf_min)

    def _warmup_base_history(self, limit: int = 200):
        """Fetch recent base timeframe klines for each symbol to reduce cold-start time."""
        interval = f"{self.base_tf_min}m"
        num_batches = RATE_LIMIT_CONFIG.get('num_batches', 4)
        batch_delay = RATE_LIMIT_CONFIG.get('batch_delay_seconds', 0.25)
        batch_size = max(1, len(self.symbols) // num_batches)
        symbol_batches = [
            self.symbols[i:i + batch_size]
            for i in range(0, len(self.symbols), batch_size)
        ]
        logger.info(f"Warmup: fetching {limit} {interval} candles per symbol in {len(symbol_batches)} batches")

        for batch_idx, batch in enumerate(symbol_batches):
            t0 = time.time()
            for symbol in batch:
                try:
                    binance_sym = convert_to_binance_symbol(symbol)
                    df = fetch_binance_klines(binance_sym, interval=interval, limit=limit)
                    if df is not None and not df.empty:
                        # Ensure columns and types match expected
                        df = df[['open','high','low','close','volume']].copy()
                        df = df.sort_index()
                        # Make index tz-naive (UTC) to match live aggregator timestamps
                        try:
                            if getattr(df.index, 'tz', None) is not None:
                                df.index = df.index.tz_convert('UTC').tz_localize(None)
                        except Exception:
                            # Fallback: try tz_localize(None) directly
                            try:
                                df.index = df.index.tz_localize(None)
                            except Exception:
                                pass
                        self.base_history[symbol] = df
                except Exception as e:
                    logger.debug(f"Warmup fetch failed for {symbol}: {e}")
            if batch_idx < len(symbol_batches) - 1:
                delay = max(0, batch_delay - (time.time() - t0))
                if delay > 0:
                    time.sleep(delay)

    def _round_qty(self, symbol: str, qty: float) -> float:
        step = QUANTITY_STEP_SIZES.get(symbol, 0.001)
        if step >= 1:
            dp = 0
        else:
            dp = len(str(step).split('.')[-1])
        scaled = int((qty + 1e-12) / step) * step
        return round(scaled, dp)

    def _round_price(self, symbol: str, price: float) -> float:
        tick = PRICE_STEP_SIZES.get(symbol, 0.01)
        if tick >= 1:
            dp = 0
        else:
            dp = len(str(tick).split('.')[-1])
        rounded = round(price / tick) * tick
        return round(rounded, dp)

    def _position_size(self, price: float) -> float:
        pct = PORTFOLIO_CONFIG.get('position_size_pct', 100.0) / 100.0
        fee = PORTFOLIO_CONFIG.get('fee_bps', 10.0) / 10000.0
        value = self.equity * pct
        value = value / (1 + fee)
        value = min(value, self.cash)
        qty = max(value / max(price, 1e-9), 0.0)
        return qty

    def _fetch_1s_tick(self, symbol: str) -> Optional[dict]:
        # Use Binance klines 1s via 1s agg if available; fallback to 1m latest and use close
        try:
            binance_sym = convert_to_binance_symbol(symbol)
            # fetch last 1 second candle
            df = fetch_binance_klines(binance_sym, interval='1s', limit=1)
            if df is None or df.empty:
                return None
            row = df.iloc[-1]
            return {
                'timestamp': df.index[-1].to_pydatetime().replace(tzinfo=None),
                'price': float(row['close']),
                'volume': float(row.get('volume', 0.0)),
            }
        except Exception as e:
            logger.debug(f"[{symbol}] fetch_1s_tick failed: {e}")
            return None

    def _compute_signal(self, symbol: str, base_df: pd.DataFrame, p: AlmaParams) -> Optional[dict]:
        # Need enough bars
        if len(base_df) < max(50, p.basis_len + p.last_n_len + 5):
            return None
        # Build alt timeframe from same 1s agg
        alt_minutes = self.base_tf_min * p.alt_multiplier
        # Build a temporary raw df by upsampling back from base (approx) for alignment
        # Better: resample original 1s to alt directly; here we re-aggregate base to alt by summing groups
        alt_df = resample_ohlcv(base_df, alt_minutes)
        # Compute ALMA-like series on base open/close and alt open/close using variant()
        close_base = base_df['close']
        open_base = base_df['open']
        close_alt = variant('ALMA', alt_df['close'], p.basis_len, p.offset_sigma, p.offset_alma).reindex(base_df.index, method='pad')
        open_alt = variant('ALMA', alt_df['open'], p.basis_len, p.offset_sigma, p.offset_alma).reindex(base_df.index, method='pad')

        # Entry condition: crossover of base close above alt ALMA on current bar
        if len(base_df) < 3:
            return None
        i = len(base_df) - 1
        prev = i - 1
        cross_up = (close_base.iloc[prev] <= close_alt.iloc[prev]) and (close_base.iloc[i] > close_alt.iloc[i])
        if not cross_up:
            return None

        # SL = lowest low of last_n_len bars excluding current
        lookback_slice = slice(max(0, i - p.last_n_len), i)
        sl = float(base_df['low'].iloc[lookback_slice].min())
        entry = float(close_base.iloc[i])
        if sl >= entry:
            return None
        risk = entry - sl
        tp = entry + p.rr_multiplier * risk
        return {
            'symbol': symbol,
            'entry_price': entry,
            'stop_loss': sl,
            'take_profit': tp,
            'timestamp': base_df.index[i],
        }

    def _place_tp_limit(self, symbol: str, qty: float, price: float) -> Optional[int]:
        price = self._round_price(symbol, price)
        qty = self._round_qty(symbol, qty)
        try:
            resp = self.roostoo.sell(symbol, qty, price=price)
            if resp.get('Success'):
                return resp.get('OrderDetail', {}).get('OrderID')
            logger.warning(f"[{symbol}] TP order failed: {resp}")
        except (RoostooAPIError, RoostooOrderError) as e:
            logger.warning(f"[{symbol}] TP order error: {e}")
        return None

    def _market_buy(self, symbol: str, qty: float) -> (Optional[int], Optional[float]):
        qty = self._round_qty(symbol, qty)
        if qty <= 0:
            return None, None
        try:
            resp = self.roostoo.buy(symbol, qty)
            if resp.get('Success'):
                od = resp.get('OrderDetail', {})
                return od.get('OrderID'), float(od.get('FilledAverPrice') or 0.0)
            logger.error(f"[{symbol}] market buy failed: {resp}")
        except (RoostooAPIError, RoostooOrderError) as e:
            logger.error(f"[{symbol}] market buy error: {e}")
        return None, None

    def _market_sell(self, symbol: str, qty: float) -> (Optional[int], Optional[float]):
        qty = self._round_qty(symbol, qty)
        if qty <= 0:
            return None, None
        try:
            resp = self.roostoo.sell(symbol, qty)
            if resp.get('Success'):
                od = resp.get('OrderDetail', {})
                return od.get('OrderID'), float(od.get('FilledAverPrice') or 0.0)
            logger.error(f"[{symbol}] market sell failed: {resp}")
        except (RoostooAPIError, RoostooOrderError) as e:
            logger.error(f"[{symbol}] market sell error: {e}")
        return None, None

    def _last_price(self, symbol: str) -> float:
        df = self.aggregators[symbol].to_df()
        return float(df['close'].iloc[-1]) if not df.empty else 0.0

    def _update_equity_cache(self):
        # Simple cash/equity tracker (can be improved by polling API for actual balances)
        self.equity = max(self.equity, self.cash)

    def run(self):
        logger.info("Starting Alma live trading loop...")
        last_base_bar_time: Dict[str, Optional[datetime]] = {s: None for s in self.symbols}
        
        # Split symbols into batches to avoid Binance 10 req/s limit
        num_batches = RATE_LIMIT_CONFIG.get('num_batches', 4)
        batch_delay = RATE_LIMIT_CONFIG.get('batch_delay_seconds', 0.25)
        batch_size = max(1, len(self.symbols) // num_batches)
        symbol_batches = [
            self.symbols[i:i + batch_size] 
            for i in range(0, len(self.symbols), batch_size)
        ]
        logger.info(f"Split {len(self.symbols)} symbols into {len(symbol_batches)} batches: {[len(b) for b in symbol_batches]}")
        logger.info(f"Using batch delay of {batch_delay}s to stay under Binance rate limits")

        while True:
            loop_start = time.time()
            
            # Fetch data in batches with delays between batches
            for batch_idx, batch in enumerate(symbol_batches):
                batch_start = time.time()
                
                for symbol in batch:
                    # 1) Fetch latest 1s tick and add to aggregator
                    tick = self._fetch_1s_tick(symbol)
                    if tick:
                        self.aggregators[symbol].add_tick(tick['price'], tick['volume'], tick['timestamp'])
                
                # Add delay between batches to ensure we stay under 10 req/s
                # Example: 5 symbols per batch with 0.25s delay = 20 req/s per batch
                # But spread across 4 batches = ~5 req/s average
                batch_elapsed = time.time() - batch_start
                if batch_idx < len(symbol_batches) - 1:  # Don't delay after last batch
                    delay = max(0, batch_delay - batch_elapsed)
                    if delay > 0:
                        time.sleep(delay)
            
            # Process all symbols for signals and position management
            for symbol in self.symbols:
                # 2) Resample to base and merge with warmup history
                live_base = self.aggregators[symbol].resample_minutes(self.base_tf_min)
                hist = self.base_history.get(symbol)
                if hist is not None and not hist.empty:
                    base_df = pd.concat([hist, live_base])
                    base_df = base_df[~base_df.index.duplicated(keep='last')].sort_index()
                    base_df = base_df.tail(1000)
                else:
                    base_df = live_base
                if base_df.empty:
                    continue
                current_bar_time = base_df.index[-1].to_pydatetime().replace(tzinfo=None)

                # 3) On a new closed base bar, compute signal and manage orders/positions
                if last_base_bar_time[symbol] is None or current_bar_time > last_base_bar_time[symbol]:
                    last_base_bar_time[symbol] = current_bar_time
                    p = self.params_map[symbol]
                    signal = self._compute_signal(symbol, base_df, p)
                    if signal and symbol not in self.positions:
                        entry_price = signal['entry_price']
                        qty = self._position_size(entry_price)
                        order_id, actual_entry = self._market_buy(symbol, qty)
                        if actual_entry is None:
                            continue
                        tp_id = self._place_tp_limit(symbol, qty, signal['take_profit'])
                        pos = Position(symbol=symbol, entry_time=datetime.utcnow(), entry_price=actual_entry,
                                       quantity=qty, stop_loss=signal['stop_loss'], take_profit=signal['take_profit'],
                                       order_id=order_id, tp_order_id=tp_id)
                        self.positions[symbol] = pos
                        # rough cash update
                        fee = actual_entry * qty * (PORTFOLIO_CONFIG.get('fee_bps', 10.0) / 10000.0)
                        self.cash -= actual_entry * qty + fee
                        # log
                        if tp_id:
                            logger.info(f"[{symbol}] Entered long via MARKET: qty={qty}, entry={actual_entry:.6f}, SL={pos.stop_loss:.6f}, TP={pos.take_profit:.6f}, TP_order={tp_id}")
                        else:
                            logger.info(f"[{symbol}] Entered long via MARKET: qty={qty}, entry={actual_entry:.6f}, SL={pos.stop_loss:.6f}, TP={pos.take_profit:.6f}")

                # 4) Manage open position: stop-loss and manual TP check (in case TP not filled)
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    last_price = self._last_price(symbol)
                    if last_price <= pos.stop_loss:
                        # Stop loss market exit
                        _, exit_price = self._market_sell(symbol, pos.quantity)
                        if exit_price is None:
                            continue
                        fee = exit_price * pos.quantity * (PORTFOLIO_CONFIG.get('fee_bps', 10.0) / 10000.0)
                        self.cash += exit_price * pos.quantity - fee
                        pnl = (exit_price - pos.entry_price) * pos.quantity - fee
                        logger.info(f"[{symbol}] STOP LOSS triggered, exit @ {exit_price:.6f}, PnL={pnl:.2f}")
                        del self.positions[symbol]
                    elif last_price >= pos.take_profit and pos.tp_order_id is None:
                        # Place TP if not already placed (safety)
                        pos.tp_order_id = self._place_tp_limit(symbol, pos.quantity, pos.take_profit)

            # end symbols loop
            self._update_equity_cache()
            # sleep remaining
            elapsed = time.time() - loop_start
            time.sleep(max(0.5, self.poll_seconds - elapsed))


if __name__ == '__main__':
    # Load symbols from live_alma_config
    from live_alma_config import TRADING_PAIRS
    
    if not TRADING_PAIRS:
        logger.error("No trading pairs configured in TRADING_PAIRS. Please set Alma_TRADING_PAIRS in .env")
        sys.exit(1)
    
    symbols = list(TRADING_PAIRS.keys())
    logger.info(f"Loaded {len(symbols)} trading pairs from config: {symbols}")
    
    trader = AlmaLiveTrader(symbols, base_tf_min=15, poll_seconds=5)
    trader.run()
