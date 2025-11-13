"""
Central configuration loader using python-dotenv.
Loads environment variables from .env at project root and exposes
typed constants for use across the codebase.
"""

import os
from typing import Optional


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse boolean-like environment variables."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# Load .env on import
try:
    from dotenv import load_dotenv
    # Resolve path to this file's directory and load ../.env if present
    _here = os.path.abspath(os.path.dirname(__file__))
    _env_path = os.path.join(_here, "..", ".env")
    load_dotenv(dotenv_path=_env_path, override=False)
except Exception:
    # Fallback to environment only if dotenv unavailable
    pass

# Roostoo API
ROOSTOO_BASE_URL: str = os.getenv("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com")
ROOSTOO_API_KEY: Optional[str] = os.getenv("api_key")
ROOSTOO_SECRET_KEY: Optional[str] = os.getenv("secret_key")

# Binance data endpoints
BINANCE_URL: str = os.getenv("BINANCE_URL", "https://data.binance.vision")
BINANCE_KLINES_PATH: str = os.getenv("BINANCE_KLINES_PATH", "/api/v3/klines")

BINANCE_PROXY_URL: Optional[str] = os.getenv("BINANCE_PROXY_URL")
BINANCE_PROXY_KLINES_PATH: str = os.getenv("BINANCE_PROXY_KLINES_PATH", "/api/klines")
USE_BINANCE_PROXY: bool = _env_flag("USE_BINANCE_PROXY", False)

# Supabase
SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
SUPABASE_KEY: Optional[str] = os.getenv("SUPABASE_KEY")
