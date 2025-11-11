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
        
        # Sync with actual account balance
        self.equity, self.cash = self._sync_account_balance()

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

    def _sync_account_balance(self) -> tuple[float, float]:
        """Sync with actual account balance from exchange"""
        try:
            balance_response = self.roostoo.get_balance()
            if not balance_response.get('Success'):
                logger.warning(f"Failed to get balance: {balance_response.get('ErrMsg')}, using default")
                default_capital = PORTFOLIO_CONFIG.get('initial_capital', 10000.0)
                return default_capital, default_capital
            
            wallet = balance_response.get('SpotWallet', {})
            usd_balance = wallet.get('USD', {})
            free_usd = float(usd_balance.get('Free', 0))
            locked_usd = float(usd_balance.get('Lock', 0))
            
            # Calculate value of existing positions
            position_value = 0.0
            for asset, balances in wallet.items():
                if asset == 'USD':
                    continue
                total_qty = float(balances.get('Free', 0)) + float(balances.get('Lock', 0))
                if total_qty > 0:
                    # We have an existing position - this shouldn't happen on fresh start
                    logger.warning(f"Found existing {asset} position: {total_qty} (will not track)")
            
            total_equity = free_usd + locked_usd + position_value
            available_cash = free_usd
            
            logger.info(f"💰 Account synced: Total Equity=${total_equity:.2f} | Available Cash=${available_cash:.2f}")
            if locked_usd > 0:
                logger.info(f"   Note: ${locked_usd:.2f} USD is locked in pending orders")
            
            return total_equity, available_cash
            
        except Exception as e:
            logger.error(f"Error syncing balance: {e}, using default")
            default_capital = PORTFOLIO_CONFIG.get('initial_capital', 10000.0)
            return default_capital, default_capital
    
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

        warmup_success = 0
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
                        warmup_success += 1
                except Exception as e:
                    logger.debug(f"Warmup fetch failed for {symbol}: {e}")
            if batch_idx < len(symbol_batches) - 1:
                delay = max(0, batch_delay - (time.time() - t0))
                if delay > 0:
                    time.sleep(delay)
        
        logger.info(f"✅ Warmup complete: {warmup_success}/{len(self.symbols)} symbols loaded with historical data")

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

    def _place_tp_limit(self, symbol: str, qty: float, price: float, max_retries: int = 3) -> Optional[int]:
        """Place take profit limit order with retry logic"""
        price = self._round_price(symbol, price)
        qty = self._round_qty(symbol, qty)
        
        if qty <= 0:
            logger.warning(f"[{symbol}] Invalid TP quantity: {qty}")
            return None
        
        for attempt in range(max_retries):
            try:
                resp = self.roostoo.sell(symbol, qty, price=price)
                if resp.get('Success'):
                    order_id = resp.get('OrderDetail', {}).get('OrderID')
                    logger.info(f"[{symbol}] TP limit order placed: ID={order_id}, qty={qty}, price={price}")
                    return order_id
                else:
                    err_msg = resp.get('ErrMsg', 'Unknown error')
                    logger.warning(f"[{symbol}] TP order failed (attempt {attempt+1}/{max_retries}): {err_msg}")
                    # Don't retry on insufficient balance errors
                    if 'insufficient balance' in err_msg.lower():
                        return None
                    time.sleep(0.5)
            except (RoostooAPIError, RoostooOrderError) as e:
                logger.warning(f"[{symbol}] TP order error (attempt {attempt+1}/{max_retries}): {e}")
                if 'insufficient balance' in str(e).lower():
                    return None
                if attempt < max_retries - 1:
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"[{symbol}] Unexpected TP order error: {e}")
                return None
        
        logger.error(f"[{symbol}] Failed to place TP order after {max_retries} attempts")
        return None
    
    def _cancel_order(self, symbol: str, order_id: int, max_retries: int = 3) -> bool:
        """Cancel an open order with retry logic. Tries by order_id, then falls back to cancel-by-pair."""
        for attempt in range(max_retries):
            try:
                # Primary: cancel by order_id
                resp = self.roostoo.cancel_order(order_id=order_id)
                if resp.get('Success'):
                    logger.info(f"[{symbol}] Cancelled order {order_id}")
                    return True
                else:
                    err_msg = resp.get('ErrMsg', 'Unknown error')
                    logger.warning(f"[{symbol}] Cancel order {order_id} failed (attempt {attempt+1}/{max_retries}): {err_msg}")
                    # If invalid id or not found, try cancel by pair to clear any hidden locks
                    if any(k in err_msg.lower() for k in ['invalid', 'not found', 'does not exist']):
                        try:
                            resp2 = self.roostoo.cancel_order(pair=symbol)
                            if resp2.get('Success'):
                                logger.info(f"[{symbol}] Fallback cancel-by-pair succeeded for TP/pendings")
                                return True
                        except Exception as e2:
                            logger.debug(f"[{symbol}] Fallback cancel-by-pair error: {e2}")
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
            except (RoostooAPIError, RoostooOrderError) as e:
                logger.warning(f"[{symbol}] Cancel order {order_id} error (attempt {attempt+1}/{max_retries}): {e}")
                # On API errors, also try fallback by pair
                try:
                    resp2 = self.roostoo.cancel_order(pair=symbol)
                    if resp2.get('Success'):
                        logger.info(f"[{symbol}] Fallback cancel-by-pair succeeded for TP/pendings")
                        return True
                except Exception:
                    pass
                if attempt < max_retries - 1:
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"[{symbol}] Unexpected cancel error: {e}")
                return False
        return False

    def _market_buy(self, symbol: str, qty: float, max_retries: int = 3) -> (Optional[int], Optional[float]):
        """Execute market buy with retry logic and validation"""
        qty = self._round_qty(symbol, qty)
        if qty <= 0:
            logger.warning(f"[{symbol}] Invalid buy quantity: {qty}")
            return None, None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"[{symbol}] Attempting market buy: qty={qty} (attempt {attempt+1}/{max_retries})")
                resp = self.roostoo.buy(symbol, qty)
                
                if resp.get('Success'):
                    od = resp.get('OrderDetail', {})
                    order_id = od.get('OrderID')
                    filled_price = float(od.get('FilledAverPrice') or 0.0)
                    
                    if filled_price > 0:
                        logger.info(f"[{symbol}] Market buy SUCCESS: ID={order_id}, qty={qty}, price={filled_price:.6f}")
                        return order_id, filled_price
                    else:
                        logger.warning(f"[{symbol}] Market buy returned 0 price, retrying...")
                        if attempt < max_retries - 1:
                            time.sleep(1.0)
                        continue
                else:
                    err_msg = resp.get('ErrMsg', 'Unknown error')
                    logger.error(f"[{symbol}] Market buy failed (attempt {attempt+1}/{max_retries}): {err_msg}")
                    
                    # Don't retry on insufficient balance
                    if 'insufficient' in err_msg.lower():
                        return None, None
                    
                    if attempt < max_retries - 1:
                        time.sleep(1.0)
                        
            except (RoostooAPIError, RoostooOrderError) as e:
                logger.error(f"[{symbol}] Market buy error (attempt {attempt+1}/{max_retries}): {e}")
                if 'insufficient' in str(e).lower():
                    return None, None
                if attempt < max_retries - 1:
                    time.sleep(1.0)
            except Exception as e:
                logger.error(f"[{symbol}] Unexpected market buy error: {e}")
                return None, None
        
        logger.error(f"[{symbol}] Market buy FAILED after {max_retries} attempts")
        return None, None

    def _market_sell(self, symbol: str, qty: float, max_retries: int = 3) -> (Optional[int], Optional[float]):
        """Execute market sell with retry logic and validation"""
        qty = self._round_qty(symbol, qty)
        if qty <= 0:
            logger.warning(f"[{symbol}] Invalid sell quantity: {qty}")
            return None, None
        
        for attempt in range(max_retries):
            try:
                logger.info(f"[{symbol}] Attempting market sell: qty={qty} (attempt {attempt+1}/{max_retries})")
                resp = self.roostoo.sell(symbol, qty)
                
                if resp.get('Success'):
                    od = resp.get('OrderDetail', {})
                    order_id = od.get('OrderID')
                    filled_price = float(od.get('FilledAverPrice') or 0.0)
                    
                    if filled_price > 0:
                        logger.info(f"[{symbol}] Market sell SUCCESS: ID={order_id}, qty={qty}, price={filled_price:.6f}")
                        return order_id, filled_price
                    else:
                        logger.warning(f"[{symbol}] Market sell returned 0 price, retrying...")
                        if attempt < max_retries - 1:
                            time.sleep(1.0)
                        continue
                else:
                    err_msg = resp.get('ErrMsg', 'Unknown error')
                    logger.error(f"[{symbol}] Market sell failed (attempt {attempt+1}/{max_retries}): {err_msg}")
                    
                    # Don't retry on insufficient balance - might need to cancel TP first
                    if 'insufficient' in err_msg.lower():
                        return None, None
                    
                    if attempt < max_retries - 1:
                        time.sleep(1.0)
                        
            except (RoostooAPIError, RoostooOrderError) as e:
                logger.error(f"[{symbol}] Market sell error (attempt {attempt+1}/{max_retries}): {e}")
                if 'insufficient' in str(e).lower():
                    return None, None
                if attempt < max_retries - 1:
                    time.sleep(1.0)
            except Exception as e:
                logger.error(f"[{symbol}] Unexpected market sell error: {e}")
                return None, None
        
        logger.error(f"[{symbol}] Market sell FAILED after {max_retries} attempts")
        return None, None

    def _last_price(self, symbol: str) -> float:
        df = self.aggregators[symbol].to_df()
        return float(df['close'].iloc[-1]) if not df.empty else 0.0

    def _update_equity_cache(self):
        # Simple cash/equity tracker (can be improved by polling API for actual balances)
        self.equity = max(self.equity, self.cash)

    def _asset_from_symbol(self, symbol: str) -> str:
        try:
            return symbol.split('/')[0]
        except Exception:
            return symbol

    def _get_asset_free_locked(self, asset: str) -> tuple[float, float]:
        try:
            bal = self.roostoo.get_balance()
            wallet = bal.get('SpotWallet', {})
            info = wallet.get(asset, {})
            free = float(info.get('Free', 0))
            locked = float(info.get('Lock', 0))
            return free, locked
        except Exception:
            return 0.0, 0.0

    def _check_tp_fill_and_close(self, symbol: str, pos: "Position") -> bool:
        """Check TP order status; if filled, close position bookkeeping and log. Returns True if closed."""
        if pos.tp_order_id is None:
            return False
        try:
            resp = self.roostoo.query_order(order_id=pos.tp_order_id)
            if not resp.get('Success'):
                return False
            orders = resp.get('OrderMatched', []) or resp.get('OrderList', [])
            if not orders:
                return False
            od = orders[0]
            status = str(od.get('Status', '')).lower()
            filled_qty = float(od.get('FilledQuantity', od.get('Quantity', 0)))
            avg_price = float(od.get('FilledAverPrice', od.get('Price', 0)) or 0.0)

            # Consider only 'filled' OR fully filled quantity as TP hit
            if status in ['filled'] or filled_qty >= pos.quantity:
                exit_price = avg_price if avg_price > 0 else self._round_price(symbol, pos.take_profit)
                fee = exit_price * pos.quantity * (PORTFOLIO_CONFIG.get('fee_bps', 10.0) / 10000.0)
                self.cash += exit_price * pos.quantity - fee
                pnl = (exit_price - pos.entry_price) * pos.quantity - fee
                pnl_pct = (pnl / (pos.entry_price * pos.quantity)) * 100 if pos.entry_price > 0 else 0.0

                # Supabase log (TP exit)
                if getattr(pos, 'trade_id', None):
                    self.supabase_logger.log_trade_exit(
                        trade_id=pos.trade_id,
                        symbol=symbol,
                        exit_time=datetime.utcnow(),
                        exit_price=exit_price,
                        quantity=pos.quantity,
                        realized_pnl=pnl,
                        realized_pnl_pct=pnl_pct,
                        exit_reason='TAKE_PROFIT',
                        fees=fee,
                        is_dry_run=False,
                        metadata={'strategy': 'ALMA', 'tp_order_id': pos.tp_order_id}
                    )
                logger.info(f"[{symbol}] TAKE PROFIT filled, exit @ {exit_price:.6f}, PnL={pnl:.2f}")
                del self.positions[symbol]
                return True

            # If order was cancelled, clear tp_order_id so we can manage manually
            if status in ['cancelled', 'canceled']:
                logger.info(f"[{symbol}] TP order {pos.tp_order_id} cancelled by exchange; will manage manually")
                pos.tp_order_id = None
                return False

            # If partially filled, we could reduce position; for simplicity, keep as is until fully filled
            return False
        except Exception as e:
            logger.debug(f"[{symbol}] TP status check failed: {e}")
            return False

    def run(self):
        logger.info("Starting Alma live trading loop...")
        last_base_bar_time: Dict[str, Optional[datetime]] = {s: None for s in self.symbols}
        last_status_log = time.time()
        
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
                    if not live_base.empty:
                        base_df = pd.concat([hist, live_base])
                        base_df = base_df[~base_df.index.duplicated(keep='last')].sort_index()
                        base_df = base_df.tail(1000)
                    else:
                        base_df = hist
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
                    
                    # Check position limits before entering
                    max_positions = PORTFOLIO_CONFIG.get('max_concurrent_positions', 3)
                    if signal and symbol not in self.positions:
                        if len(self.positions) >= max_positions:
                            logger.debug(f"[{symbol}] Signal detected but max positions ({max_positions}) reached, skipping")
                            continue
                        
                        entry_price = signal['entry_price']
                        qty = self._position_size(entry_price)
                        
                        # Check if we have enough cash
                        required_cash = entry_price * qty * (1 + PORTFOLIO_CONFIG.get('fee_bps', 10.0) / 10000.0)
                        if required_cash > self.cash:
                            logger.warning(f"[{symbol}] Insufficient cash: need ${required_cash:.2f}, have ${self.cash:.2f}")
                            continue
                        
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
                        
                        # Log to Supabase (entry)
                        trade_id = self.supabase_logger.log_trade_entry(
                            symbol=symbol,
                            entry_time=pos.entry_time,
                            entry_price=actual_entry,
                            quantity=qty,
                            stop_loss=pos.stop_loss,
                            take_profit=pos.take_profit,
                            signal_strength=100,
                            order_id=order_id,
                            is_dry_run=False,
                            metadata={
                                'strategy': 'ALMA',
                                'alt_multiplier': p.alt_multiplier,
                                'basis_len': p.basis_len,
                                'rr_multiplier': p.rr_multiplier,
                                'tp_order_id': tp_id
                            }
                        )
                        if trade_id:
                            pos.trade_id = trade_id
                        
                        # log
                        if tp_id:
                            logger.info(f"[{symbol}] Entered long via MARKET: qty={qty}, entry={actual_entry:.6f}, SL={pos.stop_loss:.6f}, TP={pos.take_profit:.6f}, TP_order={tp_id}")
                        else:
                            logger.info(f"[{symbol}] Entered long via MARKET: qty={qty}, entry={actual_entry:.6f}, SL={pos.stop_loss:.6f}, TP={pos.take_profit:.6f}")

                # 4) Manage open position: stop-loss and manual TP check (in case TP not filled)
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    # First, check if TP has been filled
                    if self._check_tp_fill_and_close(symbol, pos):
                        # Position closed; skip further checks for this symbol
                        continue
                    last_price = self._last_price(symbol)
                    if last_price <= pos.stop_loss:
                        # Stop loss market exit - cancel TP order first if it exists
                        if pos.tp_order_id is not None:
                            cancelled = self._cancel_order(symbol, pos.tp_order_id)
                            # Extra safety: cancel any remaining pending orders by pair
                            if not cancelled:
                                try:
                                    self.roostoo.cancel_order(pair=symbol)
                                except Exception:
                                    pass
                            # Clear TP id locally on successful cancel or fallback
                            pos.tp_order_id = None
                            # Wait briefly for exchange to release locks
                            time.sleep(0.5)

                        # Ensure asset is not locked before selling
                        asset = self._asset_from_symbol(symbol)
                        free, locked = self._get_asset_free_locked(asset)
                        wait_deadline = time.time() + 3.0  # wait up to 3s
                        while locked > 0 and time.time() < wait_deadline:
                            logger.debug(f"[{symbol}] Waiting for locked balance to clear: locked={locked}")
                            time.sleep(0.3)
                            free, locked = self._get_asset_free_locked(asset)

                        if locked > 0:
                            logger.error(f"[{symbol}] Cannot execute SL: balance still locked={locked}. Skipping this loop.")
                            continue

                        # Determine sellable quantity: min of position and free balance (rounded down)
                        sell_qty = self._round_qty(symbol, min(pos.quantity, max(0.0, free)))
                        if sell_qty <= 0:
                            logger.error(f"[{symbol}] Cannot execute SL: sellable qty={sell_qty}, free={free}")
                            continue

                        _, exit_price = self._market_sell(symbol, sell_qty)
                        if exit_price is None:
                            logger.error(f"[{symbol}] Failed to execute stop loss market sell for qty={sell_qty}")
                            continue

                        fee = exit_price * sell_qty * (PORTFOLIO_CONFIG.get('fee_bps', 10.0) / 10000.0)
                        self.cash += exit_price * sell_qty - fee
                        pnl = (exit_price - pos.entry_price) * sell_qty - fee
                        pnl_pct = (pnl / (pos.entry_price * sell_qty)) * 100 if pos.entry_price > 0 else 0

                        # Log to Supabase (SL exit, supports partial)
                        if getattr(pos, 'trade_id', None):
                            self.supabase_logger.log_trade_exit(
                                trade_id=pos.trade_id,
                                symbol=symbol,
                                exit_time=datetime.utcnow(),
                                exit_price=exit_price,
                                quantity=sell_qty,
                                realized_pnl=pnl,
                                realized_pnl_pct=pnl_pct,
                                exit_reason='STOP_LOSS' if sell_qty >= pos.quantity else 'STOP_LOSS_PARTIAL',
                                fees=fee,
                                is_dry_run=False,
                                metadata={'strategy': 'ALMA'}
                            )

                        if sell_qty >= pos.quantity - 1e-12:
                            logger.info(f"[{symbol}] STOP LOSS triggered, exit @ {exit_price:.6f}, qty={sell_qty}, PnL={pnl:.2f}")
                            del self.positions[symbol]
                        else:
                            # Partial exit: reduce position and continue managing remainder
                            pos.quantity = max(0.0, pos.quantity - sell_qty)
                            logger.info(f"[{symbol}] STOP LOSS partial exit @ {exit_price:.6f}, sold={sell_qty}, remaining={pos.quantity}")
                    elif last_price >= pos.take_profit and pos.tp_order_id is None:
                        # Only try to place TP once - don't retry if it failed
                        # (pos.tp_order_id remains None if placement failed initially)
                        pass

            # end symbols loop
            self._update_equity_cache()
            
            # Log position status every 5 minutes
            if time.time() - last_status_log >= 300:  # 300 seconds = 5 minutes
                last_status_log = time.time()
                if self.positions:
                    logger.info("=" * 80)
                    logger.info(f"📊 Position Status Report | Equity: ${self.equity:.2f} | Cash: ${self.cash:.2f}")
                    logger.info("=" * 80)
                    for symbol, pos in self.positions.items():
                        current_price = self._last_price(symbol)
                        pnl = (current_price - pos.entry_price) * pos.quantity if current_price > 0 else 0.0
                        pnl_pct = (pnl / (pos.entry_price * pos.quantity)) * 100 if pos.entry_price > 0 else 0.0
                        duration = datetime.utcnow() - pos.entry_time
                        logger.info(
                            f"  {symbol:12} | Entry: ${pos.entry_price:.6f} | Current: ${current_price:.6f} | "
                            f"Qty: {pos.quantity:.4f} | PnL: ${pnl:.2f} ({pnl_pct:+.2f}%) | "
                            f"SL: ${pos.stop_loss:.6f} | TP: ${pos.take_profit:.6f} | "
                            f"Duration: {str(duration).split('.')[0]}"
                        )
                    logger.info("=" * 80)
                else:
                    logger.info(f"📊 No open positions | Equity: ${self.equity:.2f} | Cash: ${self.cash:.2f}")
            
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
