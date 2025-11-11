"""
Live LSTM Trading Bot with VCRE Data Feed and Roostoo API
- Uses VCRE-style Binance US REST API polling for 15-minute candles
- LSTM model for profit opportunity prediction
- Roostoo API for order execution
- Position management with stop-loss and take-profit
"""

import sys
import os
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lstm.vcre_data_feed import VCREDataFeed
from lstm.origin.train_profit_opportunity import ProfitOpportunityPreprocessor, ProfitOpportunityLSTM
from roostoo_trading import RoostooClient, RoostooAPIError, RoostooOrderError
from utils.supabase_logger import SupabaseLogger
from utils.live_trading_config import QUANTITY_STEP_SIZES, PRICE_STEP_SIZES

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lstm_live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Open trading position"""
    symbol: str
    entry_time: datetime
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    prediction_prob: float
    order_id: Optional[int] = None
    periods_held: int = 0
    trade_id: Optional[str] = None  # Supabase trade log ID
    last_counted_candle_time: Optional[datetime] = None


class LSTMLiveTradingStrategy:
    """Live trading strategy using LSTM predictions"""
    
    def __init__(
        self,
        client: RoostooClient,
        data_feed: VCREDataFeed,
        model: torch.nn.Module,
        preprocessor: ProfitOpportunityPreprocessor,
        device: str,
        supabase_logger: SupabaseLogger,
        pair: str = "BTC/USD",
        buy_threshold: float = 0.90,
        stop_loss_pct: float = -0.01,  # -1%
        take_profit_pct: float = 0.01,  # +1%
        max_hold_periods: int = 5,  # 5 periods = 75 minutes for 15m candles
        position_size_pct: float = 0.5,
        max_capital_usd: Optional[float] = None,
        dry_run: bool = True,
        fee_bps: float = 10.0
    ):
        self.client = client
        self.data_feed = data_feed
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.supabase_logger = supabase_logger
        self.pair = pair
        self.buy_threshold = buy_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_hold_periods = max_hold_periods
        self.position_size_pct = position_size_pct
        self.max_capital_usd = max_capital_usd
        self.dry_run = dry_run
        
        # Position tracking
        self.position: Optional[Position] = None
        
        # Portfolio state
        self.equity = 0.0
        self.cash = 0.0
        self._initialize_portfolio_state()
        
        # Trade history
        self.trades = []
        
        # Extract currencies
        self.base_currency = pair.split('/')[0]
        self.quote_currency = pair.split('/')[1]
        
        # Fee configuration
        self.fee_bps = fee_bps  # 0.1% fee (10 basis points)
        
        logger.info(f"LSTMLiveTradingStrategy initialized (dry_run={dry_run})")
        logger.info(f"  Pair: {pair}")
        logger.info(f"  Buy threshold: {buy_threshold}")
        logger.info(f"  Stop loss: {stop_loss_pct*100:.1f}%")
        logger.info(f"  Take profit: {take_profit_pct*100:.1f}%")
        logger.info(f"  Max hold: {max_hold_periods} periods")
        logger.info(f"  Initial equity: ${self.equity:.2f}")
    
    def _initialize_portfolio_state(self):
        """Initialize portfolio equity and cash"""
        if self.dry_run:
            # Dry run: use configured initial capital
            # Try to get from config, otherwise use default
            try:
                from lstm.live_trading_config import TRADING_CONFIG
                initial_capital = TRADING_CONFIG.get('initial_capital', 10000.0)
            except:
                initial_capital = 10000.0
            self.cash = initial_capital
            self.equity = initial_capital
            return
        
        # Live mode: fetch from API
        cash, equity, success = self._fetch_account_equity()
        self.cash = cash
        self.equity = equity
        
        if not success:
            logger.warning("Failed to fetch account balance, using default values")
    
    def _fetch_account_equity(self) -> Tuple[float, float, bool]:
        """Fetch account equity from Roostoo API"""
        try:
            balance = self.client.get_balance()
            if not balance or not balance.get('Success'):
                return 10000.0, 10000.0, False
            
            wallet = balance.get('SpotWallet', {}) or balance.get('Wallet', {}) or {}
            usd_info = self._get_wallet_asset(wallet, 'USD')
            
            if usd_info is None:
                return 10000.0, 10000.0, False
            
            free = float(usd_info.get('Free', 0) or 0)
            locked = float(usd_info.get('Lock', 0) or 0)
            total_usd = free + locked
            
            if total_usd <= 0:
                return 10000.0, 10000.0, False
            
            return free, total_usd, True
        except Exception as e:
            logger.error(f"Error fetching account equity: {e}")
            return 10000.0, 10000.0, False
    
    def get_balance(self) -> Dict[str, float]:
        """Get current balance from Roostoo API"""
        try:
            response = self.client.get_balance()
            if response.get('Success'):
                # Try SpotWallet first (new format), then Wallet (old format)
                wallet = response.get('SpotWallet', {}) or response.get('Wallet', {}) or {}
                
                # Get base currency balance
                base_info = self._get_wallet_asset(wallet, self.base_currency)
                base_balance = float(base_info.get('Free', 0) or 0) if base_info else 0.0
                
                # Get quote currency balance
                quote_info = self._get_wallet_asset(wallet, self.quote_currency)
                quote_balance = float(quote_info.get('Free', 0) or 0) if quote_info else 0.0
                
                return {
                    'base': base_balance,
                    'quote': quote_balance
                }
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
        return {'base': 0.0, 'quote': 0.0}
    
    def _get_wallet_asset(self, wallet: dict, asset: str) -> Optional[dict]:
        """Get asset info from wallet with case-insensitive lookup"""
        for key in (asset, asset.upper(), asset.lower()):
            if key in wallet:
                return wallet[key]
        return None
    
    def predict(self, df: pd.DataFrame) -> float:
        """Generate prediction for the latest data point"""
        try:
            # Prepare data (use fitted scaler)
            X, _ = self.preprocessor.prepare_data(df, fit_scaler=False)
            
            if len(X) == 0:
                logger.warning("No valid sequences for prediction")
                return 0.0
            
            # Get last sequence
            x = torch.tensor(X[-1:], dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                logit = self.model(x)
                prob = torch.sigmoid(logit).cpu().item()
            
            return prob
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return 0.0
    
    def check_entry_signal(self, current_price: float) -> Optional[Dict]:
        """Check for entry signal based on LSTM prediction"""
        if self.position is not None:
            return None
        
        # Get latest data (exclude partial candle for stable entry decisions)
        df = self.data_feed.get_latest_data(lookback_periods=100, include_partial=False)
        
        if df.empty or len(df) < self.preprocessor.sequence_length:
            logger.warning("Insufficient data for prediction")
            return None
        
        # Generate prediction
        prob = self.predict(df)
        
        print(f"🤖 Model prediction: {prob:.4f}")
        logger.info(f"🤖 Model prediction: {prob:.4f}")
        
        # Check if signal meets threshold
        if prob >= self.buy_threshold:
            # Calculate stop loss and take profit
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
            
            signal = {
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'prediction_prob': prob,
                'timestamp': datetime.now()
            }
            
            print(f"\n✅ Buy signal! (prob {prob:.4f} >= threshold {self.buy_threshold:.2f})")
            print(f"   Entry: ${current_price:,.2f}")
            print(f"   Stop Loss: ${stop_loss:,.2f} ({self.stop_loss_pct*100:.1f}%)")
            print(f"   Take Profit: ${take_profit:,.2f} ({self.take_profit_pct*100:.1f}%)")
            logger.info(f"✅ Buy signal! (prob {prob:.4f} >= threshold {self.buy_threshold:.2f})")
            logger.info(f"   Entry: ${current_price:,.2f}")
            logger.info(f"   Stop Loss: ${stop_loss:,.2f} ({self.stop_loss_pct*100:.1f}%)")
            logger.info(f"   Take Profit: ${take_profit:,.2f} ({self.take_profit_pct*100:.1f}%)")
            
            return signal
        else:
            logger.info(f"⏸️  No signal (prob {prob:.4f} < threshold {self.buy_threshold:.2f})")
            return None
    
    def check_exit_signal(self, current_price: float, current_high: float = None, current_low: float = None) -> Optional[str]:
        """
        Check if position should be closed.
        Uses candle high/low for more accurate stop loss/take profit detection.
        
        Args:
            current_price: Current close price
            current_high: Current candle high (optional, defaults to current_price)
            current_low: Current candle low (optional, defaults to current_price)
        
        Returns:
            Exit reason ('STOP_LOSS', 'TAKE_PROFIT', 'MAX_HOLD') or None
        """
        if self.position is None:
            return None
        
        # Use high/low if provided, otherwise use current price
        high = current_high if current_high is not None else current_price
        low = current_low if current_low is not None else current_price
        
        # Check stop loss (use low price - worst case)
        if low <= self.position.stop_loss:
            return 'STOP_LOSS'
        
        # Check take profit (use high price - best case)
        if high >= self.position.take_profit:
            return 'TAKE_PROFIT'
        
        # Check max hold time
        if self.position.periods_held >= self.max_hold_periods:
            return 'MAX_HOLD'
        
        return None
    
    def round_quantity(self, quantity: float) -> float:
        """Round quantity to proper step size for the exchange"""
        # Use per-symbol step size when available (fallback to 0.001)
        step_size = QUANTITY_STEP_SIZES.get(self.pair, 0.001)
        
        # Determine decimal places based on step size
        if step_size >= 1:
            decimal_places = 0
        else:
            decimal_places = len(str(step_size).split('.')[-1])
        
        # Round down to nearest step size
        # Floor division approach to avoid over-sizing
        try:
            factor = int(round(1 / step_size)) if step_size < 1 else 1
            rounded = (int(quantity * factor) / factor)
        except Exception:
            # Fallback method similar to VCRE implementation
            scale = 10 ** decimal_places
            rounded = int(quantity * scale / (step_size * scale)) * step_size
        
        # Normalize floating point precision
        rounded = round(rounded, decimal_places)
        return rounded
    
    def round_price(self, price: float) -> float:
        """Round price to proper tick size for the exchange"""
        # Use per-symbol tick size when available (fallback to $0.01)
        tick_size = PRICE_STEP_SIZES.get(self.pair, 0.01)
        
        # Determine decimal places for normalization
        if tick_size >= 1:
            decimal_places = 0
        else:
            decimal_places = len(str(tick_size).split('.')[-1])
        
        # Round to nearest tick size
        rounded = round(price / tick_size) * tick_size
        rounded = round(rounded, decimal_places)
        return rounded
    
    def calculate_position_size(self, entry_price: float) -> float:
        """Calculate position size based on portfolio allocation"""
        # Use percentage of current equity
        position_value = self.equity * self.position_size_pct
        
        # Account for fees (pre-deduct)
        fee_rate = self.fee_bps / 10000
        position_value = position_value / (1 + fee_rate)
        
        # Apply max capital limit if set
        if self.max_capital_usd is not None:
            position_value = min(position_value, float(self.max_capital_usd))
        
        # Don't exceed available cash (keep small buffer)
        min_buffer = 10.0  # Keep at least $10
        position_value = min(position_value, self.cash - min_buffer)
        
        if position_value <= 0:
            logger.warning(f"Insufficient cash: ${self.cash:.2f}")
            return 0.0
        
        # Calculate quantity
        quantity = position_value / entry_price
        
        # Round to proper step size
        quantity = self.round_quantity(quantity)
        
        return quantity
    
    def _deduct_entry_cost(self, quantity: float, price: float):
        """Deduct entry cost including fees from cash"""
        fee = quantity * price * (self.fee_bps / 10000)
        self.cash -= (quantity * price + fee)
        logger.debug(f"Cash after entry: ${self.cash:.2f} (fee: ${fee:.2f})")
    
    def _add_exit_proceeds(self, quantity: float, price: float, fee: float):
        """Add exit proceeds minus fees to cash"""
        proceeds = quantity * price
        self.cash += (proceeds - fee)
        logger.debug(f"Cash after exit: ${self.cash:.2f} (proceeds: ${proceeds:.2f}, fee: ${fee:.2f})")
    
    def execute_entry(self, signal: Dict) -> bool:
        """Execute entry order"""
        entry_price = signal['entry_price']
        quantity = self.calculate_position_size(entry_price)
        
        if quantity <= 0:
            logger.warning("Position size too small or insufficient balance")
            return False
        
        try:
            print(f"\n📊 Placing BUY order:")
            print(f"   Price: ${entry_price:,.2f}")
            print(f"   Quantity: {quantity} {self.base_currency}")
            print(f"   Value: ${entry_price * quantity:,.2f}")
            print(f"   Probability: {signal['prediction_prob']:.4f}")
            logger.info(f"\n📊 Placing BUY order:")
            logger.info(f"   Price: ${entry_price:,.2f}")
            logger.info(f"   Quantity: {quantity} {self.base_currency}")
            logger.info(f"   Value: ${entry_price * quantity:,.2f}")
            logger.info(f"   Probability: {signal['prediction_prob']:.4f}")
            
            if not self.dry_run:
                # Place market buy order
                response = self.client.buy(self.pair, quantity)
                
                if response.get('Success'):
                    order_detail = response.get('OrderDetail', {})
                    actual_entry = order_detail.get('FilledAverPrice', entry_price)
                    order_id = order_detail.get('OrderID')
                    
                    print(f"✅ Buy order executed! Order ID: {order_id}")
                    print(f"   Filled at: ${actual_entry:,.2f}")
                    logger.info(f"✅ Buy order executed! Order ID: {order_id}")
                    logger.info(f"   Filled at: ${actual_entry:,.2f}")
                else:
                    print(f"❌ Buy order failed: {response.get('ErrMsg')}")
                    logger.error(f"❌ Buy order failed: {response.get('ErrMsg')}")
                    return False
            else:
                # Dry run mode
                actual_entry = entry_price
                order_id = None
                logger.info(f"✅ [DRY RUN] Would buy {quantity} {self.base_currency} @ ${entry_price:,.2f}")
            
            # Create position
            entry_time = datetime.now()
            self.position = Position(
                symbol=self.pair,
                entry_time=entry_time,
                entry_price=actual_entry,
                quantity=quantity,
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit'],
                prediction_prob=signal['prediction_prob'],
                order_id=order_id,
                periods_held=0
            )
            # Initialize last_counted_candle_time to the latest CLOSED candle at entry
            try:
                df_closed = self.data_feed.get_latest_data(lookback_periods=1, include_partial=False)
                if not df_closed.empty:
                    last_ts = pd.to_datetime(df_closed['timestamp'].iloc[-1]).to_pydatetime().replace(tzinfo=None)
                    self.position.last_counted_candle_time = last_ts
            except Exception as _:
                pass
            
            # Log to Supabase
            trade_id = self.supabase_logger.log_trade_entry(
                symbol=self.pair,
                entry_time=entry_time,
                entry_price=actual_entry,
                quantity=quantity,
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit'],
                signal_strength=int(signal['prediction_prob'] * 100),
                order_id=order_id,
                is_dry_run=self.dry_run,
                metadata={
                    'strategy': 'LSTM',
                    'model_threshold': self.buy_threshold,
                    'timeframe_minutes': self.data_feed.timeframe_minutes,
                    'position_size_pct': self.position_size_pct,
                    'max_capital_usd': self.max_capital_usd
                }
            )
            # Store trade_id in position for later reference
            if trade_id:
                self.position.trade_id = trade_id
            
            # Update cash (deduct entry cost + fees)
            self._deduct_entry_cost(quantity, actual_entry)
            
            # Update equity
            self.equity = self.cash + (quantity * actual_entry)
            
            # Record trade
            self.trades.append({
                'type': 'BUY',
                'price': actual_entry,
                'quantity': quantity,
                'value': actual_entry * quantity,
                'timestamp': self.position.entry_time.isoformat(),
                'prob': signal['prediction_prob'],
                'order_id': order_id
            })
            
            return True
            
        except (RoostooAPIError, RoostooOrderError) as e:
            logger.error(f"❌ Error placing buy order: {e}")
            self._log_order_error('execute_entry', e, {'signal': signal})
            return False
    
    def execute_exit(self, current_price: float, reason: str) -> bool:
        """Execute exit order"""
        if self.position is None:
            logger.warning("No position to exit")
            return False

        try:
            if reason == 'TAKE_PROFIT':
                exit_price = self.position.take_profit
            elif reason == 'STOP_LOSS':
                exit_price = self.position.stop_loss
            else:
                exit_price = current_price
            
            # Calculate P&L
            pnl_pct = (exit_price - self.position.entry_price) / self.position.entry_price * 100
            pnl_value = (exit_price - self.position.entry_price) * self.position.quantity
            
            print(f"\n📊 Placing SELL order:")
            print(f"   Exit Price: ${exit_price:,.2f}")
            print(f"   Current Price: ${current_price:,.2f}")
            print(f"   Quantity: {self.position.quantity} {self.base_currency}")
            print(f"   Reason: {reason}")
            print(f"   PnL: {pnl_pct:+.2f}% (${pnl_value:+,.2f})")
            print(f"   Hold duration: {self.position.periods_held} periods")
            logger.info(f"\n📊 Placing SELL order:")
            logger.info(f"   Exit Price: ${exit_price:,.2f}")
            logger.info(f"   Current Price: ${current_price:,.2f}")
            logger.info(f"   Quantity: {self.position.quantity} {self.base_currency}")
            logger.info(f"   Reason: {reason}")
            logger.info(f"   PnL: {pnl_pct:+.2f}% (${pnl_value:+,.2f})")
            logger.info(f"   Hold duration: {self.position.periods_held} periods")
            
            if not self.dry_run:
                # Place market sell order
                response = self.client.sell(self.pair, self.position.quantity)
                
                if response.get('Success'):
                    order_detail = response.get('OrderDetail', {})
                    actual_exit = order_detail.get('FilledAverPrice', current_price)
                    order_id = order_detail.get('OrderID')
                    
                    # Recalculate P&L with actual exit price
                    pnl_pct = (actual_exit - self.position.entry_price) / self.position.entry_price * 100
                    pnl_value = (actual_exit - self.position.entry_price) * self.position.quantity
                    
                    print(f"✅ Sell order executed! Order ID: {order_id}")
                    print(f"   Filled at: ${actual_exit:,.2f}")
                    print(f"   Final PnL: {pnl_pct:+.2f}% (${pnl_value:+,.2f})")
                    logger.info(f"✅ Sell order executed! Order ID: {order_id}")
                    logger.info(f"   Filled at: ${actual_exit:,.2f}")
                    logger.info(f"   Final PnL: {pnl_pct:+.2f}% (${pnl_value:+,.2f})")
                else:
                    print(f"❌ Sell order failed: {response.get('ErrMsg')}")
                    logger.error(f"❌ Sell order failed: {response.get('ErrMsg')}")
                    return False
            else:
                # Dry run mode
                actual_exit = current_price
                order_id = None
                logger.info(f"✅ [DRY RUN] Would sell {self.position.quantity} {self.base_currency} @ ${current_price:,.2f}")
            
            # Calculate fees
            fee_rate = self.fee_bps / 10000
            fee = (self.position.entry_price * self.position.quantity + actual_exit * self.position.quantity) * fee_rate
            net_pnl = pnl_value - fee
            
            # Update cash (add exit proceeds - fees)
            self._add_exit_proceeds(self.position.quantity, actual_exit, fee)
            
            # Update equity (now all in cash)
            self.equity = self.cash
            
            # Log to Supabase
            if hasattr(self.position, 'trade_id') and self.position.trade_id:
                self.supabase_logger.log_trade_exit(
                    trade_id=self.position.trade_id,
                    symbol=self.pair,
                    exit_time=datetime.now(),
                    exit_price=actual_exit,
                    quantity=self.position.quantity,
                    realized_pnl=net_pnl,
                    realized_pnl_pct=pnl_pct,
                    exit_reason=reason,
                    fees=fee,
                    is_dry_run=self.dry_run,
                    metadata={
                        'strategy': 'LSTM',
                        'periods_held': self.position.periods_held,
                        'entry_prob': self.position.prediction_prob
                    }
                )
            
            # Record trade
            self.trades.append({
                'type': 'SELL',
                'price': actual_exit,
                'quantity': self.position.quantity,
                'timestamp': datetime.now().isoformat(),
                'reason': reason,
                'pnl_pct': pnl_pct,
                'pnl_value': pnl_value,
                'order_id': order_id,
                'periods_held': self.position.periods_held
            })
            
            # Clear position
            self.position = None
            
            return True
            
        except (RoostooAPIError, RoostooOrderError) as e:
            logger.error(f"❌ Error placing sell order: {e}")
            self._log_order_error('execute_exit', e, {'reason': reason, 'price': current_price})
            return False
    
    def _log_order_error(self, function_name: str, error: Exception, details: dict = None):
        """Log order execution error to logger and Supabase"""
        logger.error(f"[{self.pair}] Error in {function_name}: {error}")
        # Ensure strategy tag is present in error details
        _err_details = (details or {}).copy()
        _err_details.setdefault('strategy', 'LSTM')
        self.supabase_logger.log_error(
            error_type='ORDER_ERROR',
            severity='ERROR',
            error_message=str(error),
            symbol=self.pair,
            function_name=function_name,
            is_dry_run=self.dry_run,
            error_details=_err_details
        )
    
    def get_trade_summary(self) -> Dict[str, Any]:
        """Get summary of all trades"""
        sells = [t for t in self.trades if t['type'] == 'SELL']
        
        if not sells:
            return {
                'total_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_pnl_pct': 0.0
            }
        
        pnls = [t['pnl_value'] for t in sells]
        pnl_pcts = [t['pnl_pct'] for t in sells]
        winning = [p for p in pnls if p > 0]
        
        return {
            'total_trades': len(sells),
            'total_pnl': sum(pnls),
            'win_rate': len(winning) / len(pnls) if pnls else 0.0,
            'avg_pnl_pct': np.mean(pnl_pcts) if pnl_pcts else 0.0,
            'winning_trades': len(winning),
            'losing_trades': len(pnls) - len(winning)
        }


class LSTMLiveTrader:
    """Main live trading bot orchestrator"""
    
    def __init__(
        self,
        model_path: str,
        training_data_path: str,
        api_key: str,
        secret_key: str,
        base_url: str = "https://mock-api.roostoo.com",
        pair: str = "BTC/USD",
        buy_threshold: float = 0.90,
        timeframe_minutes: int = 15,
        check_interval: int = 60,
        position_size_pct: float = 0.5,
        dry_run: bool = True,
        max_capital_usd: Optional[float] = None,
        enable_supabase: bool = True,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        stop_loss_pct: float = -0.01,
        take_profit_pct: float = 0.01,
        max_hold_periods: int = 5,
        fee_bps: float = 10.0
    ):
        self.model_path = model_path
        self.training_data_path = training_data_path
        self.pair = pair
        self.buy_threshold = buy_threshold
        self.timeframe_minutes = timeframe_minutes
        self.check_interval = check_interval
        self.dry_run = dry_run
        
        # Initialize Roostoo client
        self.client = RoostooClient(api_key, secret_key, base_url)
        
        # Initialize data feed
        self.data_feed = VCREDataFeed(
            symbol=pair,
            timeframe_minutes=timeframe_minutes,
            max_candles=300
        )
        
        # Load model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        self.preprocessor = ProfitOpportunityPreprocessor(
            sequence_length=60,
            forecast_periods=5,
            profit_threshold=0.002,
            verbose=False  # Disable verbose output during live trading
        )
        
        self.model = None
        self._load_model()
        
        # Initialize Supabase logger
        self.supabase_logger = SupabaseLogger(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            enabled=enable_supabase
        )
        
        # Initialize strategy
        self.strategy = LSTMLiveTradingStrategy(
            client=self.client,
            data_feed=self.data_feed,
            model=self.model,
            preprocessor=self.preprocessor,
            device=self.device,
            supabase_logger=self.supabase_logger,
            pair=pair,
            buy_threshold=buy_threshold,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_hold_periods=max_hold_periods,
            position_size_pct=position_size_pct,
            dry_run=dry_run,
            max_capital_usd=max_capital_usd,
            fee_bps=fee_bps
        )
        
        self.running = False
        self.last_checked_closed_ts: Optional[datetime] = None
    
    def _load_model(self):
        """Load trained LSTM model"""
        logger.info(f"\n📦 Loading model from: {self.model_path}")
        
        # Fit preprocessor on training data
        df = pd.read_csv(self.training_data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Fitting preprocessor on {len(df)} data points...")
        # Temporarily enable verbose for initial fitting
        self.preprocessor.verbose = True
        X, y = self.preprocessor.prepare_data(df, fit_scaler=True)
        # Disable verbose for live trading
        self.preprocessor.verbose = False
        
        # Load model
        self.model = ProfitOpportunityLSTM(
            input_dim=self.preprocessor.get_feature_dim(),
            hidden_dim=96,
            dropout=0.2
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        
        logger.info(f"✅ Model loaded successfully!")
    
    def warmup(self, num_candles: int = 300):
        """Load historical data for warmup"""
        self.data_feed.load_historical_warmup(num_candles)
    
    def run_once(self):
        """Run one iteration of the trading loop"""
        from datetime import timezone, timedelta
        
        # Get current time in both UTC and HKT
        now_utc = datetime.now(timezone.utc)
        hkt = timezone(timedelta(hours=8))
        now_hkt = now_utc.astimezone(hkt)
        
        print("\n" + "=" * 70)
        print(f"⏰ {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC ({now_hkt.strftime('%H:%M:%S')} HKT)")
        print("=" * 70)
        logger.info("\n" + "=" * 70)
        logger.info(f"⏰ {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        logger.info("=" * 70)
        
        # Update latest candles
        self.data_feed.update_latest_candles(num_candles=10)
        
        # Get current candle data (including high/low)
        print(f"DEBUG: Candle buffer size: {len(self.data_feed.candle_buffer.candles)}")
        if len(self.data_feed.candle_buffer.candles) > 0:
            last_candle = self.data_feed.candle_buffer.candles[-1]
            last_candle_hkt = last_candle.timestamp.replace(tzinfo=timezone.utc).astimezone(hkt)
            print(f"DEBUG: Last candle in buffer: {last_candle.timestamp} UTC ({last_candle_hkt.strftime('%Y-%m-%d %H:%M:%S')} HKT)")
        if len(self.data_feed.candle_buffer.candles) == 0:
            logger.warning("No candle data available")
            print("WARNING: No candle data available")
            return
        
        current_candle = self.data_feed.candle_buffer.candles[-1]
        current_price = current_candle.close
        current_high = current_candle.high
        current_low = current_candle.low
        
        print(f"\n💰 Current {self.pair} candle:")
        print(f"   Close: ${current_price:,.2f}")
        print(f"   High: ${current_high:,.2f}")
        print(f"   Low: ${current_low:,.2f}")
        logger.info(f"\n💰 Current {self.pair} candle:")
        logger.info(f"   Close: ${current_price:,.2f}")
        logger.info(f"   High: ${current_high:,.2f}")
        logger.info(f"   Low: ${current_low:,.2f}")
        
        # Check balance
        balance = self.strategy.get_balance()
        print(f"💼 Balance: {balance['base']:.6f} {self.strategy.base_currency}, "
                   f"${balance['quote']:,.2f} {self.strategy.quote_currency}")
        logger.info(f"💼 Balance: {balance['base']:.6f} {self.strategy.base_currency}, "
                   f"${balance['quote']:,.2f} {self.strategy.quote_currency}")
        
        # Check if we have a position
        if self.strategy.position is not None:
            # Increment periods_held only when a NEW closed candle appears
            try:
                df_closed = self.data_feed.get_latest_data(lookback_periods=1, include_partial=False)
                if not df_closed.empty:
                    last_closed_ts = pd.to_datetime(df_closed['timestamp'].iloc[-1]).to_pydatetime().replace(tzinfo=None)
                    if (self.strategy.position.last_counted_candle_time is None or
                        last_closed_ts > self.strategy.position.last_counted_candle_time):
                        self.strategy.position.periods_held += 1
                        self.strategy.position.last_counted_candle_time = last_closed_ts
            except Exception as _:
                # If any issue reading closed candle timestamp, skip increment this loop
                pass
            
            pnl_pct = (current_price - self.strategy.position.entry_price) / self.strategy.position.entry_price * 100
            
            print(f"\n📈 Current Position:")
            print(f"   Entry: ${self.strategy.position.entry_price:,.2f}")
            print(f"   Current: ${current_price:,.2f}")
            print(f"   PnL: {pnl_pct:+.2f}%")
            print(f"   Periods held: {self.strategy.position.periods_held}/{self.strategy.max_hold_periods}")
            print(f"   Stop Loss: ${self.strategy.position.stop_loss:,.2f}")
            print(f"   Take Profit: ${self.strategy.position.take_profit:,.2f}")
            logger.info(f"\n📈 Current Position:")
            logger.info(f"   Entry: ${self.strategy.position.entry_price:,.2f}")
            logger.info(f"   Current: ${current_price:,.2f}")
            logger.info(f"   PnL: {pnl_pct:+.2f}%")
            logger.info(f"   Periods held: {self.strategy.position.periods_held}/{self.strategy.max_hold_periods}")
            logger.info(f"   Stop Loss: ${self.strategy.position.stop_loss:,.2f}")
            logger.info(f"   Take Profit: ${self.strategy.position.take_profit:,.2f}")
            
            # Check exit conditions (pass high/low for accurate detection)
            exit_reason = self.strategy.check_exit_signal(current_price, current_high, current_low)
            
            if exit_reason:
                print(f"\n🚨 Exit signal: {exit_reason}")
                logger.info(f"\n🚨 Exit signal: {exit_reason}")
                self.strategy.execute_exit(current_price, exit_reason)
        else:
            try:
                df_closed = self.data_feed.get_latest_data(lookback_periods=1, include_partial=False)
                print(f"DEBUG: df_closed shape: {df_closed.shape}, empty: {df_closed.empty}")
                if not df_closed.empty:
                    print(f"DEBUG: df_closed last timestamp: {df_closed['timestamp'].iloc[-1]}")
                    last_closed_ts = pd.to_datetime(df_closed['timestamp'].iloc[-1]).to_pydatetime().replace(tzinfo=None)
                    should_check = (self.last_checked_closed_ts is None) or (last_closed_ts > self.last_checked_closed_ts)
                    
                    # Convert to HKT for display (UTC+8)
                    from datetime import timezone, timedelta
                    hkt = timezone(timedelta(hours=8))
                    last_closed_hkt = last_closed_ts.replace(tzinfo=timezone.utc).astimezone(hkt)
                    last_checked_hkt = self.last_checked_closed_ts.replace(tzinfo=timezone.utc).astimezone(hkt) if self.last_checked_closed_ts else None
                    
                    print(f"DEBUG: should_check={should_check}")
                    print(f"       Last closed candle: {last_closed_ts} UTC ({last_closed_hkt.strftime('%Y-%m-%d %H:%M:%S')} HKT)")
                    if last_checked_hkt:
                        print(f"       Last checked: {self.last_checked_closed_ts} UTC ({last_checked_hkt.strftime('%Y-%m-%d %H:%M:%S')} HKT)")
                    
                    if should_check:
                        print("DEBUG: Checking entry signal...")
                        signal = self.strategy.check_entry_signal(current_price)
                        if signal:
                            self.strategy.execute_entry(signal)
                        self.last_checked_closed_ts = last_closed_ts
                    else:
                        # Show prediction even when waiting for new candle
                        # Get latest data (exclude partial candle for stable predictions)
                        df = self.data_feed.get_latest_data(lookback_periods=100, include_partial=False)
                        
                        if not df.empty and len(df) >= self.strategy.preprocessor.sequence_length:
                            # Generate prediction
                            prob = self.strategy.predict(df)
                            print(f"🤖 Model prediction: {prob:.4f}")
                            logger.info(f"🤖 Model prediction: {prob:.4f}")
                        
                        # Compute next candle close from current time (next timeframe boundary)
                        period = self.timeframe_minutes
                        current_period_start = now_utc.replace(minute=(now_utc.minute // period) * period, second=0, microsecond=0)
                        next_candle_utc = current_period_start + timedelta(minutes=period)
                        next_candle_hkt = next_candle_utc.replace(tzinfo=timezone.utc).astimezone(hkt)
                        print(f"\n⏸️  Waiting for new candle")
                        print(f"   Next candle closes at: {next_candle_utc.strftime('%H:%M')} UTC ({next_candle_hkt.strftime('%H:%M')} HKT)")
                        logger.info(f"\n⏸️  Waiting for new candle (next: {next_candle_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC)")
            except Exception as e:
                logger.error(f"Error checking entry signal: {e}")
        
        # Print trade summary
        summary = self.strategy.get_trade_summary()
        if summary['total_trades'] > 0:
            logger.info(f"\n📊 Trade Summary:")
            logger.info(f"   Total trades: {summary['total_trades']}")
            logger.info(f"   Win rate: {summary['win_rate']*100:.1f}%")
            logger.info(f"   Avg PnL: {summary['avg_pnl_pct']:+.2f}%")
            logger.info(f"   Total PnL: ${summary['total_pnl']:+,.2f}")
    
    def run(self, duration_hours: Optional[int] = None):
        """
        Run the trading bot continuously.
        
        Args:
            duration_hours: Run for specified hours (None = run indefinitely)
        """
        logger.info("\n" + "=" * 70)
        logger.info("🤖 LSTM LIVE TRADING BOT")
        logger.info("=" * 70)
        logger.info(f"\nConfiguration:")
        logger.info(f"  Pair: {self.pair}")
        logger.info(f"  Timeframe: {self.timeframe_minutes} minutes")
        logger.info(f"  Buy threshold: {self.buy_threshold}")
        logger.info(f"  Check interval: {self.check_interval}s")
        logger.info(f"  Dry run: {self.dry_run}")
        if duration_hours:
            logger.info(f"  Duration: {duration_hours} hours")
        
        self.running = True
        start_time = datetime.now()
        
        try:
            while self.running:
                # Check if duration limit reached
                if duration_hours:
                    elapsed = (datetime.now() - start_time).total_seconds() / 3600
                    if elapsed >= duration_hours:
                        logger.info(f"\n⏰ Duration limit reached ({duration_hours} hours)")
                        break
                
                # Run one iteration
                self.run_once()
                
                # Wait for next check
                logger.info(f"\n⏳ Waiting {self.check_interval} seconds...")
                time.sleep(self.check_interval)
            
            logger.info("\n" + "=" * 70)
            logger.info("✅ Trading bot stopped")
            logger.info("=" * 70)
            
            # Final summary
            summary = self.strategy.get_trade_summary()
            logger.info(f"\n📊 Final Summary:")
            logger.info(f"   Total trades: {summary['total_trades']}")
            if summary['total_trades'] > 0:
                logger.info(f"   Win rate: {summary['win_rate']*100:.1f}%")
                logger.info(f"   Avg PnL: {summary['avg_pnl_pct']:+.2f}%")
                logger.info(f"   Total PnL: ${summary['total_pnl']:+,.2f}")
            
            # Save trade log
            if self.strategy.trades:
                log_file = f"lstm_trade_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(log_file, 'w') as f:
                    json.dump(self.strategy.trades, f, indent=2)
                logger.info(f"\n💾 Trade log saved to: {log_file}")
        
        except KeyboardInterrupt:
            logger.info("\n\n⚠️  Bot stopped by user")
            
            # Emergency: close any open positions
            if self.strategy.position is not None and not self.dry_run:
                logger.info("\n🚨 Closing open position...")
                current_price = self.data_feed.get_current_price()
                if current_price:
                    self.strategy.execute_exit(current_price, 'EMERGENCY_STOP')
        
        except Exception as e:
            logger.error(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
