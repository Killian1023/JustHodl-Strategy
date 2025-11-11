"""
Central configuration loader using python-dotenv.
Loads environment variables from .env at project root and exposes
typed constants for use across the codebase.
"""

import os
from typing import Optional

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

# Supabase
SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
SUPABASE_KEY: Optional[str] = os.getenv("SUPABASE_KEY")
