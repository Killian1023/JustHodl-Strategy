"""
Roostoo API Configuration
Store your API credentials here
"""

from typing import Optional

try:
    # Import central config which loads .env via python-dotenv
    from utils.config import (
        ROOSTOO_BASE_URL,
        ROOSTOO_API_KEY,
        ROOSTOO_SECRET_KEY,
    )
except Exception:
    # Fallbacks if central config import fails
    import os
    ROOSTOO_BASE_URL = os.getenv("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com")
    ROOSTOO_API_KEY = os.getenv("api_key")
    ROOSTOO_SECRET_KEY = os.getenv("secret_key")

ROOSTOO_CONFIG = {
    "base_url": ROOSTOO_BASE_URL,
    "api_key": ROOSTOO_API_KEY,
    "secret_key": ROOSTOO_SECRET_KEY,
}
