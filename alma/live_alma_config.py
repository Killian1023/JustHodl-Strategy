"""
Live configuration for Alma strategy (ALMA + RR TP/SL)
- Defines AlmaParams
- Portfolio, risk, logging configs
- Quantity/price step sizes
- Helpers to build per-coin params from optimization results (optimization_results_alma)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
import json
import os


# ========= Alma strategy parameters =========
@dataclass
class AlmaParams:
    # Core timeframe
    base_tf_min: int = 15

    # Strategy core (fixed in Alma): ALMA
    alt_multiplier: int = 8          # optimized
    basis_len: int = 2               # optimized
    offset_sigma: int = 5            # optimized
    offset_alma: float = 0.85        # optimized

    # Risk model
    last_n_len: int = 10             # optimized
    rr_multiplier: float = 2.0       # optimized
    
    # ATR Swing Stop parameters
    atr_len: int = 14                # ATR period (optimized)
    atr_mult: float = 1.0            # ATR multiplier for stop buffer (optimized)


# ========= Portfolio & risk =========
PORTFOLIO_CONFIG = {
    "initial_capital": 50000.0,
    "max_concurrent_positions": 3,
    "position_size_pct": 18,
    "fee_bps": 10.0,
    "clear_account_on_start": True,
    "clear_account_on_end": True,
}

RISK_CONFIG = {
    "max_risk_per_trade_pct": 10.0,
    "min_position_value_usd": 10.0,
}

# Timeframe for Alma (minutes)
TIMEFRAME_MINUTES = 15

# ========= Rounding settings (copied from VCRE live config) =========
QUANTITY_STEP_SIZES: Dict[str, float] = {
    "1000CHEEMS/USD": 1.0,
    "AAVE/USD": 0.001,
    "ADA/USD": 0.1,
    "APT/USD": 0.01,
    "ARB/USD": 0.1,
    "ASTER/USD": 0.01,
    "AVAX/USD": 0.01,
    "AVNT/USD": 0.1,
    "BIO/USD": 0.1,
    "BMT/USD": 0.1,
    "BNB/USD": 0.001,
    "BONK/USD": 1.0,
    "BTC/USD": 0.00001,
    "CAKE/USD": 0.01,
    "CFX/USD": 1.0,
    "CRV/USD": 0.1,
    "DOGE/USD": 1.0,
    "DOT/USD": 0.01,
    "EDEN/USD": 0.1,
    "EIGEN/USD": 0.01,
    "ENA/USD": 0.01,
    "ETH/USD": 0.0001,
    "FET/USD": 0.1,
    "FIL/USD": 0.01,
    "FLOKI/USD": 1.0,
    "FORM/USD": 0.1,
    "HBAR/USD": 1.0,
    "HEMI/USD": 0.1,
    "ICP/USD": 0.01,
    "LINEA/USD": 1.0,
    "LINK/USD": 0.01,
    "LISTA/USD": 0.1,
    "LTC/USD": 0.001,
    "MIRA/USD": 0.1,
    "NEAR/USD": 0.1,
    "OMNI/USD": 0.01,
    "ONDO/USD": 0.1,
    "OPEN/USD": 0.1,
    "PAXG/USD": 0.0001,
    "PENDLE/USD": 0.1,
    "PENGU/USD": 1.0,
    "PEPE/USD": 1.0,
    "PLUME/USD": 1.0,
    "POL/USD": 0.1,
    "PUMP/USD": 1.0,
    "S/USD": 0.1,
    "SEI/USD": 0.1,
    "SHIB/USD": 1.0,
    "SOL/USD": 0.001,
    "SOMI/USD": 0.1,
    "STO/USD": 0.1,
    "SUI/USD": 0.1,
    "TAO/USD": 0.0001,
    "TON/USD": 0.01,
    "TRUMP/USD": 0.001,
    "TRX/USD": 0.1,
    "TUT/USD": 1.0,
    "UNI/USD": 0.01,
    "VIRTUAL/USD": 0.1,
    "WIF/USD": 0.01,
    "WLD/USD": 0.1,
    "WLFI/USD": 0.1,
    "XLM/USD": 1.0,
    "XPL/USD": 0.1,
    "XRP/USD": 0.1,
    "ZEC/USD": 0.001,
    "ZEN/USD": 0.01,
}

PRICE_STEP_SIZES: Dict[str, float] = {
    "1000CHEEMS/USD": 0.000001,
    "AAVE/USD": 0.01,
    "ADA/USD": 0.0001,
    "APT/USD": 0.001,
    "ARB/USD": 0.0001,
    "ASTER/USD": 0.001,
    "AVAX/USD": 0.01,
    "AVNT/USD": 0.0001,
    "BIO/USD": 0.0001,
    "BMT/USD": 0.0001,
    "BNB/USD": 0.01,
    "BONK/USD": 0.00000001,
    "BTC/USD": 0.01,
    "CAKE/USD": 0.001,
    "CFX/USD": 0.0001,
    "CRV/USD": 0.0001,
    "DOGE/USD": 0.00001,
    "DOT/USD": 0.001,
    "EDEN/USD": 0.0001,
    "EIGEN/USD": 0.001,
    "ENA/USD": 0.0001,
    "ETH/USD": 0.01,
    "FET/USD": 0.0001,
    "FIL/USD": 0.001,
    "FLOKI/USD": 0.00000001,
    "FORM/USD": 0.0001,
    "HBAR/USD": 0.00001,
    "HEMI/USD": 0.0001,
    "ICP/USD": 0.001,
    "LINEA/USD": 0.00001,
    "LINK/USD": 0.01,
    "LISTA/USD": 0.0001,
    "LTC/USD": 0.01,
    "MIRA/USD": 0.0001,
    "NEAR/USD": 0.001,
    "OMNI/USD": 0.01,
    "ONDO/USD": 0.0001,
    "OPEN/USD": 0.0001,
    "PAXG/USD": 0.01,
    "PENDLE/USD": 0.001,
    "PENGU/USD": 0.000001,
    "PEPE/USD": 0.00000001,
    "PLUME/USD": 0.00001,
    "POL/USD": 0.0001,
    "PUMP/USD": 0.000001,
    "S/USD": 0.0001,
    "SEI/USD": 0.0001,
    "SHIB/USD": 0.00000001,
    "SOL/USD": 0.01,
    "SOMI/USD": 0.0001,
    "STO/USD": 0.0001,
    "SUI/USD": 0.0001,
    "TAO/USD": 0.1,
    "TON/USD": 0.001,
    "TRUMP/USD": 0.001,
    "TRX/USD": 0.0001,
    "TUT/USD": 0.00001,
    "UNI/USD": 0.001,
    "VIRTUAL/USD": 0.0001,
    "WIF/USD": 0.001,
    "WLD/USD": 0.001,
    "WLFI/USD": 0.0001,
    "XLM/USD": 0.0001,
    "XPL/USD": 0.0001,
    "XRP/USD": 0.0001,
    "ZEC/USD": 0.01,
    "ZEN/USD": 0.001,
}

# ========= Logging =========
LOG_CONFIG = {
    "log_file": "live_trading.log",
    "log_level": "INFO",
    "log_trades": True,
    "log_signals": True,
}

# ========= Rate Limiting =========
RATE_LIMIT_CONFIG = {
    "num_batches": 4,  # Split symbols into this many batches
    "batch_delay_seconds": 0.5,  # Delay between batches to avoid rate limits
}


# ========= Load trading pairs from environment =========
def load_trading_pairs_from_env() -> Dict[str, AlmaParams]:
    """
    Load trading pair parameters from environment variables.
    Expected format in .env:
    Alma_TRADING_PAIRS="BTC_USD,ETH_USD,..."
    BTC_USD_ALT_MULTIPLIER=15
    BTC_USD_BASIS_LEN=1
    ...
    """
    pairs_str = os.getenv('ALMA_TRADING_PAIRS', '')
    if not pairs_str:
        return {}
    
    trading_pairs = {}
    for pair_key in pairs_str.split(','):
        pair_key = pair_key.strip()
        if not pair_key:
            continue
        
        # Convert BTC_USD to BTC/USD
        pair = pair_key.replace('_', '/')
        
        # Read parameters from environment
        try:
            alt_multiplier = int(os.getenv(f'{pair_key}_ALT_MULTIPLIER', 8))
            basis_len = int(os.getenv(f'{pair_key}_BASIS_LEN', 2))
            offset_sigma = int(os.getenv(f'{pair_key}_OFFSET_SIGMA', 5))
            offset_alma = float(os.getenv(f'{pair_key}_OFFSET_ALMA', 0.85))
            last_n_len = int(os.getenv(f'{pair_key}_LAST_N_LEN', 10))
            rr_multiplier = float(os.getenv(f'{pair_key}_RR_MULTIPLIER', 2.0))
            atr_len = int(os.getenv(f'{pair_key}_ATR_LEN', 14))
            atr_mult = float(os.getenv(f'{pair_key}_ATR_MULT', 1.0))
            
            trading_pairs[pair] = AlmaParams(
                base_tf_min=TIMEFRAME_MINUTES,
                alt_multiplier=alt_multiplier,
                basis_len=basis_len,
                offset_sigma=offset_sigma,
                offset_alma=offset_alma,
                last_n_len=last_n_len,
                rr_multiplier=rr_multiplier,
                atr_len=atr_len,
                atr_mult=atr_mult,
            )
        except (ValueError, TypeError) as e:
            print(f"Warning: Failed to load parameters for {pair}: {e}")
            continue
    
    return trading_pairs

# Load trading pairs from environment variables
TRADING_PAIRS: Dict[str, AlmaParams] = load_trading_pairs_from_env()

# Fallback: Manual per-coin Alma parameters (used if env vars not set)
if not TRADING_PAIRS:
    TRADING_PAIRS = {
        "BTC/USD": AlmaParams(
            base_tf_min=TIMEFRAME_MINUTES,
            alt_multiplier=15,
            basis_len=1,
            offset_sigma=2,
            offset_alma=0.32,
            last_n_len=28,
            rr_multiplier=1.25,
            atr_len=14,
            atr_mult=1.0,
        ),
        "ETH/USD": AlmaParams(
            base_tf_min=TIMEFRAME_MINUTES,
            alt_multiplier=12,
            basis_len=1,
            offset_sigma=3,
            offset_alma=0.28,
            last_n_len=25,
            rr_multiplier=1.30,
            atr_len=14,
            atr_mult=1.0,
        ),
    }

RATE_LIMIT_CONFIG = {
    "num_batches": 4,  # Split symbols into this many batches
    "batch_delay_seconds": 0.25,  # Delay between batches to avoid rate limits
}