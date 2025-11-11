"""Custom exceptions for Roostoo API client."""


class RoostooAPIError(Exception):
    """Base exception for Roostoo API errors."""
    pass


class RoostooAuthError(RoostooAPIError):
    """Exception raised for authentication errors."""
    pass


class RoostooOrderError(RoostooAPIError):
    """Exception raised for order-related errors."""
    pass
