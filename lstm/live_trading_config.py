"""
Configuration for LSTM Live Trading Bot
"""

import os

# Trading Configuration
TRADING_CONFIG = {
    # Trading pair (Roostoo format)
    'pair': 'BTC/USD',
    
    # Model prediction threshold for buy signal
    'buy_threshold': 0.85,
    
    # Risk management
    'stop_loss_pct': -0.01,      # -0.1% stop loss (use -0.01 for -1%)
    'take_profit_pct': 0.01,     # +0.1% take profit (use 0.01 for +1%)
    'max_hold_periods': 5,       # Max 5 periods (75 minutes for 15m candles)
    
    # Position sizing
    'position_size_pct': 0.5,   # Use 50% of equity per trade (use 0.85 for 85%)
    'max_capital_usd': None,   # Hard cap: max $1000 per trade (None = no limit)
    'initial_capital': 10000.0,  # Initial capital for dry run mode
    
    # Fees
    'fee_bps': 10.0,             # Trading fee in basis points (10 = 0.1%)
    
    # Data feed
    'timeframe_minutes': 15,     # 15-minute candles (must match model training)
    'warmup_candles': 300,       # Number of historical candles to load
    
    # Polling interval
    'check_interval': 30,        # Check every 60 seconds
    
    # Dry run mode
    'dry_run': False,             # Set to False for real trading
}

# Model Configuration
MODEL_CONFIG = {
    # Model architecture (must match training)
    'sequence_length': 60,
    'forecast_periods': 5,
    'profit_threshold': 0.002,
    'hidden_dim': 96,
    'dropout': 0.2,
}

# File Paths (relative to lstm directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILE_PATHS = {
    # Trained model file
    'model_path': os.path.join(BASE_DIR, 'profit_opportunity_best.pth'),
    
    # Training data for fitting preprocessor scaler
    'training_data_path': os.path.join(BASE_DIR, 'binance_BTCUSDT_15m_2023-01-01_to_2025-09-30.csv'),
    
    # Log file
    'log_file': 'lstm_live_trading.log',
}

# Logging Configuration
LOG_CONFIG = {
    'log_level': 'INFO',
    'log_file': FILE_PATHS['log_file'],
    'log_trades': True,
    'log_signals': True,
}

# Supabase Configuration
SUPABASE_CONFIG = {
    'enabled': True,  # Enable Supabase logging
    # Credentials will be read from environment variables:
    # - SUPABASE_URL
    # - SUPABASE_KEY
    # Or you can set them here (not recommended for production):
    # 'supabase_url': 'your_supabase_url',
    # 'supabase_key': 'your_supabase_key',
}
