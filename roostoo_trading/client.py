"""Main Roostoo API client for trading operations."""

import time
import hmac
import hashlib
from typing import Optional, Dict, Any, Union
import requests

from .exceptions import RoostooAPIError, RoostooAuthError, RoostooOrderError


class RoostooClient:
    """
    Roostoo Trading API Client.
    
    This client provides methods for trading operations including:
    - Placing orders (market and limit)
    - Querying orders
    - Canceling orders
    - Getting account balance
    - Checking pending orders
    
    Args:
        api_key: Your Roostoo API key
        secret_key: Your Roostoo secret key
        base_url: API base URL (default: https://mock-api.roostoo.com)
    """
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = "https://mock-api.roostoo.com"
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip('/')
        
    def _get_timestamp(self) -> str:
        """Generate 13-digit millisecond timestamp."""
        return str(int(time.time() * 1000))
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """
        Generate HMAC SHA256 signature for signed endpoints.
        
        Args:
            params: Dictionary of parameters to sign
            
        Returns:
            Hexadecimal signature string
        """
        # Sort parameters by key and create query string
        sorted_keys = sorted(params.keys())
        total_params = "&".join(f"{k}={params[k]}" for k in sorted_keys)
        
        # Generate HMAC SHA256 signature
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            total_params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _get_signed_headers(self, payload: Dict[str, Any]) -> tuple:
        """
        Generate signed headers for RCL_TopLevelCheck endpoints.
        
        Args:
            payload: Request payload dictionary
            
        Returns:
            Tuple of (headers, payload, total_params_string)
        """
        # Add timestamp to payload
        payload['timestamp'] = self._get_timestamp()
        
        # Generate signature
        signature = self._generate_signature(payload)
        
        # Create headers
        headers = {
            'RST-API-KEY': self.api_key,
            'MSG-SIGNATURE': signature
        }
        
        # Create total params string for POST body
        sorted_keys = sorted(payload.keys())
        total_params = "&".join(f"{k}={payload[k]}" for k in sorted_keys)
        
        return headers, payload, total_params
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Handle API response and raise appropriate exceptions.
        
        Args:
            response: requests Response object
            
        Returns:
            JSON response data
            
        Raises:
            RoostooAPIError: For API errors
        """
        try:
            response.raise_for_status()
            data = response.json()
            
            # Check if API returned an error
            if isinstance(data, dict) and not data.get('Success', True):
                error_msg = data.get('ErrMsg', 'Unknown error')
                raise RoostooAPIError(f"API Error: {error_msg}")
            
            return data
            
        except requests.exceptions.HTTPError as e:
            raise RoostooAPIError(f"HTTP Error: {e}")
        except requests.exceptions.RequestException as e:
            raise RoostooAPIError(f"Request Error: {e}")
        except ValueError as e:
            raise RoostooAPIError(f"Invalid JSON response: {e}")
    
    # ==================== Account Methods ====================
    
    def get_balance(self) -> Dict[str, Any]:
        """
        Get current wallet balance.
        
        Returns:
            Dictionary containing wallet balances for all assets
            
        Example:
            {
                "Success": true,
                "Wallet": {
                    "BTC": {"Free": 0.454878, "Lock": 0.555},
                    "USD": {"Free": 98389854.15, "Lock": 1601798.20}
                }
            }
        """
        url = f"{self.base_url}/v3/balance"
        headers, payload, _ = self._get_signed_headers({})
        
        response = requests.get(url, headers=headers, params=payload)
        return self._handle_response(response)
    
    def get_pending_count(self) -> Dict[str, Any]:
        """
        Get count of pending orders.
        
        Returns:
            Dictionary with total pending count and breakdown by pair
            
        Example:
            {
                "Success": true,
                "TotalPending": 3,
                "OrderPairs": {"BAT/USD": 1, "LINK/USD": 2}
            }
        """
        url = f"{self.base_url}/v3/pending_count"
        headers, payload, _ = self._get_signed_headers({})
        
        response = requests.get(url, headers=headers, params=payload)
        return self._handle_response(response)
    
    # ==================== Trading Methods ====================
    
    def place_order(
        self,
        pair: str,
        side: str,
        quantity: Union[float, int],
        order_type: str = "MARKET",
        price: Optional[Union[float, int]] = None
    ) -> Dict[str, Any]:
        """
        Place a new order.
        
        Args:
            pair: Trading pair (e.g., "BTC/USD", "ETH/USD")
            side: Order side - "BUY" or "SELL"
            quantity: Order quantity
            order_type: Order type - "MARKET" or "LIMIT" (default: "MARKET")
            price: Limit price (required for LIMIT orders)
            
        Returns:
            Dictionary containing order details
            
        Raises:
            RoostooOrderError: If order parameters are invalid
            
        Example:
            # Market order
            client.place_order("BTC/USD", "BUY", 0.01)
            
            # Limit order
            client.place_order("BTC/USD", "BUY", 0.01, "LIMIT", 50000)
        """
        # Validate parameters
        side = side.upper()
        order_type = order_type.upper()
        
        if side not in ["BUY", "SELL"]:
            raise RoostooOrderError(f"Invalid side: {side}. Must be 'BUY' or 'SELL'")
        
        if order_type not in ["MARKET", "LIMIT"]:
            raise RoostooOrderError(f"Invalid order type: {order_type}. Must be 'MARKET' or 'LIMIT'")
        
        if order_type == "LIMIT" and price is None:
            raise RoostooOrderError("LIMIT orders require a price")
        
        # Normalize pair format (add /USD if not present)
        if "/" not in pair:
            pair = f"{pair}/USD"
        
        # Build payload
        payload = {
            'pair': pair,
            'side': side,
            'type': order_type,
            'quantity': str(quantity)
        }
        
        if order_type == "LIMIT":
            payload['price'] = str(price)
        
        # Make request
        url = f"{self.base_url}/v3/place_order"
        headers, _, total_params = self._get_signed_headers(payload)
        headers['Content-Type'] = 'application/x-www-form-urlencoded'
        
        response = requests.post(url, headers=headers, data=total_params)
        return self._handle_response(response)
    
    def buy(
        self,
        pair: str,
        quantity: Union[float, int],
        price: Optional[Union[float, int]] = None
    ) -> Dict[str, Any]:
        """
        Convenience method to place a BUY order.
        
        Args:
            pair: Trading pair (e.g., "BTC/USD")
            quantity: Order quantity
            price: Limit price (if None, places market order)
            
        Returns:
            Dictionary containing order details
        """
        order_type = "LIMIT" if price is not None else "MARKET"
        return self.place_order(pair, "BUY", quantity, order_type, price)
    
    def sell(
        self,
        pair: str,
        quantity: Union[float, int],
        price: Optional[Union[float, int]] = None
    ) -> Dict[str, Any]:
        """
        Convenience method to place a SELL order.
        
        Args:
            pair: Trading pair (e.g., "BTC/USD")
            quantity: Order quantity
            price: Limit price (if None, places market order)
            
        Returns:
            Dictionary containing order details
        """
        order_type = "LIMIT" if price is not None else "MARKET"
        return self.place_order(pair, "SELL", quantity, order_type, price)
    
    def query_order(
        self,
        order_id: Optional[int] = None,
        pair: Optional[str] = None,
        pending_only: Optional[bool] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Query order history or specific orders.
        
        Args:
            order_id: Specific order ID to query (if provided, other params ignored)
            pair: Filter by trading pair
            pending_only: If True, only return pending orders
            limit: Maximum number of orders to return (default: 100)
            
        Returns:
            Dictionary containing matched orders
            
        Example:
            # Query specific order
            client.query_order(order_id=123)
            
            # Query all pending orders for a pair
            client.query_order(pair="BTC/USD", pending_only=True)
            
            # Query all orders
            client.query_order()
        """
        url = f"{self.base_url}/v3/query_order"
        payload = {}
        
        if order_id is not None:
            payload['order_id'] = str(order_id)
        elif pair is not None:
            payload['pair'] = pair
        
        if pending_only is not None:
            payload['pending_only'] = 'TRUE' if pending_only else 'FALSE'
        
        if limit is not None:
            payload['limit'] = str(limit)
        
        headers, _, total_params = self._get_signed_headers(payload)
        headers['Content-Type'] = 'application/x-www-form-urlencoded'
        
        response = requests.post(url, headers=headers, data=total_params)
        return self._handle_response(response)
    
    def cancel_order(
        self,
        order_id: Optional[int] = None,
        pair: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cancel pending orders.
        
        Args:
            order_id: Specific order ID to cancel
            pair: Cancel all pending orders for this pair
            
        Note:
            - If neither order_id nor pair is provided, cancels ALL pending orders
            - Only pending orders can be canceled
            - Only one of order_id or pair should be provided
            
        Returns:
            Dictionary containing list of canceled order IDs
            
        Example:
            # Cancel specific order
            client.cancel_order(order_id=123)
            
            # Cancel all pending orders for a pair
            client.cancel_order(pair="BTC/USD")
            
            # Cancel all pending orders
            client.cancel_order()
        """
        url = f"{self.base_url}/v3/cancel_order"
        payload = {}
        
        if order_id is not None:
            payload['order_id'] = str(order_id)
        elif pair is not None:
            payload['pair'] = pair
        
        headers, _, total_params = self._get_signed_headers(payload)
        headers['Content-Type'] = 'application/x-www-form-urlencoded'
        
        response = requests.post(url, headers=headers, data=total_params)
        return self._handle_response(response)
    
    def cancel_all_orders(self, pair: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel all pending orders, optionally filtered by pair.
        
        Args:
            pair: If provided, only cancel orders for this pair
            
        Returns:
            Dictionary containing list of canceled order IDs
        """
        return self.cancel_order(pair=pair)
