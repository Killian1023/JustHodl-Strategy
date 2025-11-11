"""
Roostoo Trading API Client
A Python package for trading with the Roostoo API.
"""

from .client import RoostooClient
from .exceptions import RoostooAPIError, RoostooAuthError, RoostooOrderError

__version__ = "1.0.0"
__all__ = ["RoostooClient", "RoostooAPIError", "RoostooAuthError", "RoostooOrderError"]
