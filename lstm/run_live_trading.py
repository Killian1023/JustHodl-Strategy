#!/usr/bin/env python3
"""
Launch script for LSTM Live Trading Bot

Usage:
    python run_live_trading.py [--duration HOURS] [--live]
    
Options:
    --duration HOURS    Run for specified hours (default: run indefinitely)
    --live              Enable live trading mode (default: dry run)
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lstm.live_lstm_trader import LSTMLiveTrader
from lstm.live_trading_config import TRADING_CONFIG, MODEL_CONFIG, FILE_PATHS, SUPABASE_CONFIG
from roostoo_trading.config import ROOSTOO_CONFIG


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='LSTM Live Trading Bot')
    parser.add_argument('--duration', type=float, default=None,
                       help='Run for specified hours (default: run indefinitely)')
    parser.add_argument('--live', action='store_true',
                       help='Enable live trading mode (default: dry run)')
    args = parser.parse_args()
    
    # Use config default; --live overrides to False
    dry_run = TRADING_CONFIG['dry_run']
    if args.live:
        dry_run = False
    
    print("=" * 70)
    print("LSTM LIVE TRADING BOT - ROOSTOO API")
    print("=" * 70)
    
    if dry_run:
        print("\n⚠️  DRY RUN MODE - No real orders will be placed")
    else:
        print("\n🚨 LIVE MODE - Real orders will be placed!")
    
    # Validate files exist
    model_path = FILE_PATHS['model_path']
    data_path = FILE_PATHS['training_data_path']
    
    if not os.path.exists(model_path):
        print(f"\n❌ Model file not found: {model_path}")
        print("   Please ensure the model file exists or update FILE_PATHS in live_trading_config.py")
        return
    
    if not os.path.exists(data_path):
        print(f"\n❌ Training data file not found: {data_path}")
        print("   Please ensure the CSV exists or update FILE_PATHS in live_trading_config.py")
        return
    
    # Display configuration
    print(f"\n📋 Configuration:")
    print(f"   Pair: {TRADING_CONFIG['pair']}")
    print(f"   Timeframe: {TRADING_CONFIG['timeframe_minutes']} minutes")
    print(f"   Buy threshold: {TRADING_CONFIG['buy_threshold']}")
    print(f"   Stop loss: {TRADING_CONFIG['stop_loss_pct']*100:.1f}%")
    print(f"   Take profit: {TRADING_CONFIG['take_profit_pct']*100:.1f}%")
    print(f"   Max hold: {TRADING_CONFIG['max_hold_periods']} periods")
    print(f"   Position size: {TRADING_CONFIG['position_size_pct']*100:.0f}% of balance")
    if TRADING_CONFIG['max_capital_usd']:
        print(f"   Max capital: ${TRADING_CONFIG['max_capital_usd']:,.0f}")
    print(f"   Check interval: {TRADING_CONFIG['check_interval']}s")
    if args.duration:
        print(f"   Duration: {args.duration} hours")
    print(f"   Dry run: {dry_run}")
    print(f"   Supabase logging: {'Enabled' if SUPABASE_CONFIG['enabled'] else 'Disabled'}")
    
    # Initialize bot
    print(f"\n🤖 Initializing bot...")
    bot = LSTMLiveTrader(
        model_path=model_path,
        training_data_path=data_path,
        api_key=ROOSTOO_CONFIG['api_key'],
        secret_key=ROOSTOO_CONFIG['secret_key'],
        base_url=ROOSTOO_CONFIG['base_url'],
        pair=TRADING_CONFIG['pair'],
        buy_threshold=TRADING_CONFIG['buy_threshold'],
        timeframe_minutes=TRADING_CONFIG['timeframe_minutes'],
        check_interval=TRADING_CONFIG['check_interval'],
        position_size_pct=TRADING_CONFIG['position_size_pct'],
        dry_run=dry_run,
        max_capital_usd=TRADING_CONFIG['max_capital_usd'],
        enable_supabase=SUPABASE_CONFIG['enabled'],
        supabase_url=SUPABASE_CONFIG.get('supabase_url'),
        supabase_key=SUPABASE_CONFIG.get('supabase_key'),
        stop_loss_pct=TRADING_CONFIG['stop_loss_pct'],
        take_profit_pct=TRADING_CONFIG['take_profit_pct'],
        max_hold_periods=TRADING_CONFIG['max_hold_periods'],
        fee_bps=TRADING_CONFIG.get('fee_bps', 10.0)
    )
    
    # Load historical data for warmup
    print(f"\n📊 Loading historical data...")
    bot.warmup(num_candles=TRADING_CONFIG['warmup_candles'])
    
    # Check if we have enough data
    if not bot.data_feed.has_enough_data(min_candles=MODEL_CONFIG['sequence_length']):
        print(f"\n❌ Insufficient data: need at least {MODEL_CONFIG['sequence_length']} candles")
        return
    
    # Display data info
    info = bot.data_feed.get_data_info()
    print(f"\n✅ Data feed ready:")
    print(f"   Total candles: {info['total_candles']}")
    print(f"   Last candle: {info['last_candle_time']}")
    print(f"   Current price: ${info['current_price']:,.2f}")
    
    # Start trading
    print(f"\n🚀 Starting trading bot...")
    print(f"   Press Ctrl+C to stop\n")
    
    bot.run(duration_hours=args.duration)


if __name__ == '__main__':
    main()
