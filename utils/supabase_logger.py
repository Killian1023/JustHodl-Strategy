"""
Supabase Logger for Trade and Error Logging
Logs trades and errors to Supabase database in web3_hackathon schema
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import uuid4
import traceback
from utils.config import SUPABASE_URL as CFG_SUPABASE_URL, SUPABASE_KEY as CFG_SUPABASE_KEY

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logging.warning("Supabase client not installed. Install with: pip install supabase")


logger = logging.getLogger(__name__)


class SupabaseLogger:
    """Logger for trades and errors to Supabase database"""
    
    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None, enabled: bool = True):
        """
        Initialize Supabase logger
        
        Args:
            supabase_url: Supabase project URL (or set SUPABASE_URL env var)
            supabase_key: Supabase anon/service key (or set SUPABASE_KEY env var)
            enabled: Whether logging is enabled
        """
        self.enabled = enabled and SUPABASE_AVAILABLE
        self.client: Optional[Client] = None
        
        if not self.enabled:
            if not SUPABASE_AVAILABLE:
                logger.warning("SupabaseLogger disabled: supabase package not installed")
            else:
                logger.info("SupabaseLogger disabled by configuration")
            return
        
        # Get credentials from args or environment
        url = supabase_url or CFG_SUPABASE_URL
        key = supabase_key or CFG_SUPABASE_KEY
        
        if not url or not key:
            logger.error("Supabase credentials not provided. Set SUPABASE_URL and SUPABASE_KEY environment variables")
            self.enabled = False
            return
        
        try:
            # Create client
            self.client = create_client(url, key)
            # Store schema name for table access
            self.schema = 'web3_hackathon'
            logger.info(f"✅ SupabaseLogger initialized successfully (schema: {self.schema})")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            self.enabled = False
    
    def log_trade_entry(
        self,
        symbol: str,
        entry_time: datetime,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        signal_strength: int,
        order_id: Optional[int] = None,
        is_dry_run: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Log a trade entry
        
        Returns:
            trade_id (UUID string) if successful, None otherwise
        """
        if not self.enabled or not self.client:
            return None
        
        trade_id = str(uuid4())
        
        try:
            data = {
                'trade_id': trade_id,
                'symbol': symbol,
                'action': 'ENTRY',
                'entry_time': entry_time.isoformat(),
                'entry_price': float(entry_price),
                'quantity': float(quantity),
                'original_quantity': float(quantity),
                'stop_loss_price': float(stop_loss),
                'take_profit_price': float(take_profit),
                'signal_strength': signal_strength,
                'order_id': order_id,
                'is_dry_run': is_dry_run,
                'metadata': metadata or {}
            }
            
            result = self.client.schema(self.schema).table('trade_logs').insert(data).execute()
            logger.debug(f"[{symbol}] Trade entry logged to Supabase: {trade_id}")
            return trade_id
            
        except Exception as e:
            logger.error(f"Failed to log trade entry to Supabase: {e}")
            return None
    
    def log_trade_exit(
        self,
        trade_id: str,
        symbol: str,
        exit_time: datetime,
        exit_price: float,
        quantity: float,
        realized_pnl: float,
        realized_pnl_pct: float,
        exit_reason: str,
        fees: float = 0.0,
        is_dry_run: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log a trade exit (full or partial)
        
        Note: Each exit gets a unique log_id, but references the same trade_id
        This allows multiple exits (partial TP + final exit) for the same trade
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.client:
            return False
        
        try:
            # Determine action type based on exit reason
            if 'Partial' in exit_reason or 'partial' in exit_reason.lower():
                action = 'PARTIAL_TP'
            elif 'Stop Loss' in exit_reason:
                action = 'STOP_LOSS'
            elif 'Trailing' in exit_reason:
                action = 'TRAILING_STOP'
            else:
                action = 'EXIT'
            
            # Generate unique log_id for this exit (allows multiple exits per trade)
            log_id = str(uuid4())
            
            data = {
                'log_id': log_id,  # Unique ID for this exit log
                'trade_id': trade_id,  # Reference to original trade entry
                'symbol': symbol,
                'action': action,
                'exit_time': exit_time.isoformat(),
                'exit_price': float(exit_price),
                'quantity': float(quantity),
                'realized_pnl': float(realized_pnl),
                'realized_pnl_pct': float(realized_pnl_pct),
                'exit_reason': exit_reason,
                'fees': float(fees),
                'is_dry_run': is_dry_run,
                'metadata': metadata or {}
            }
            
            result = self.client.schema(self.schema).table('trade_logs').insert(data).execute()
            logger.debug(f"[{symbol}] Trade exit logged to Supabase: {exit_reason} (log_id={log_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log trade exit to Supabase: {e}")
            return False
    
    def update_trade_position(
        self,
        trade_id: str,
        unrealized_pnl: Optional[float] = None,
        quantity: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing trade position (for tracking unrealized P&L, partial exits, etc.)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.client:
            return False
        
        try:
            update_data = {}
            
            if unrealized_pnl is not None:
                update_data['unrealized_pnl'] = float(unrealized_pnl)
            
            if quantity is not None:
                update_data['quantity'] = float(quantity)
            
            if metadata is not None:
                update_data['metadata'] = metadata
            
            if not update_data:
                return True  # Nothing to update
            
            result = self.client.schema(self.schema).table('trade_logs').update(update_data).eq('trade_id', trade_id).execute()
            return True
            
        except Exception as e:
            logger.error(f"Failed to update trade position in Supabase: {e}")
            return False
    
    def log_error(
        self,
        error_type: str,
        severity: str,
        error_message: str,
        symbol: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None,
        function_name: Optional[str] = None,
        is_dry_run: bool = False,
        include_traceback: bool = True
    ) -> Optional[str]:
        """
        Log an error to Supabase
        
        Args:
            error_type: Type of error (e.g., 'API_ERROR', 'ORDER_ERROR', 'STRATEGY_ERROR')
            severity: Severity level ('INFO', 'WARNING', 'ERROR', 'CRITICAL')
            error_message: Human-readable error message
            symbol: Trading symbol if applicable
            error_details: Additional error context
            function_name: Name of function where error occurred
            is_dry_run: Whether this occurred in dry-run mode
            include_traceback: Whether to include stack trace
        
        Returns:
            error_id (UUID string) if successful, None otherwise
        """
        if not self.enabled or not self.client:
            return None
        
        error_id = str(uuid4())
        
        try:
            details = error_details or {}
            
            # Add traceback if requested
            if include_traceback:
                details['traceback'] = traceback.format_exc()
            
            data = {
                'error_id': error_id,
                'error_type': error_type,
                'severity': severity.upper(),
                'symbol': symbol,
                'error_message': error_message,
                'error_details': details,
                'function_name': function_name,
                'is_dry_run': is_dry_run
            }
            
            result = self.client.schema(self.schema).table('error_logs').insert(data).execute()
            logger.debug(f"Error logged to Supabase: {error_type} - {error_message}")
            return error_id
            
        except Exception as e:
            logger.error(f"Failed to log error to Supabase: {e}")
            return None
    
    def get_recent_trades(self, symbol: Optional[str] = None, limit: int = 100) -> list:
        """
        Retrieve recent trades from database
        
        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum number of records to return
        
        Returns:
            List of trade records
        """
        if not self.enabled or not self.client:
            return []
        
        try:
            query = self.client.schema(self.schema).table('trade_logs').select('*').order('created_at', desc=True).limit(limit)
            
            if symbol:
                query = query.eq('symbol', symbol)
            
            result = query.execute()
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Failed to retrieve trades from Supabase: {e}")
            return []
    
    def get_recent_errors(self, severity: Optional[str] = None, limit: int = 100) -> list:
        """
        Retrieve recent errors from database
        
        Args:
            severity: Filter by severity level (optional)
            limit: Maximum number of records to return
        
        Returns:
            List of error records
        """
        if not self.enabled or not self.client:
            return []
        
        try:
            query = self.client.schema(self.schema).table('error_logs').select('*').order('created_at', desc=True).limit(limit)
            
            if severity:
                query = query.eq('severity', severity.upper())
            
            result = query.execute()
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Failed to retrieve errors from Supabase: {e}")
            return []
