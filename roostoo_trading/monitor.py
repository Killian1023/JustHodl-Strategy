"""
Monitor Roostoo account status
Check balance, pending orders, and recent order history
"""

import os
import sys
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roostoo_trading import RoostooClient, RoostooAPIError
from roostoo_trading.config import ROOSTOO_CONFIG


def main():
    """Monitor trading account status"""
    
    # Get API credentials (env vars or config file)
    api_key = os.getenv('ROOSTOO_API_KEY') or ROOSTOO_CONFIG.get('api_key')
    secret_key = os.getenv('ROOSTOO_SECRET_KEY') or ROOSTOO_CONFIG.get('secret_key')
    
    if not api_key or not secret_key:
        print("Error: Set credentials in roostoo_trading/config.py or environment variables")
        sys.exit(1)
    
    # Initialize client
    client = RoostooClient(api_key=api_key, secret_key=secret_key)
    
    print("=" * 60)
    print(f"Roostoo Account Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Get balance
        print("\n📊 Account Balance:")
        balance = client.get_balance()
        
        if balance.get('Success'):
            wallet = balance.get('Wallet', {})
            
            for asset, amounts in wallet.items():
                free = amounts.get('Free', 0)
                locked = amounts.get('Lock', 0)
                total = free + locked
                
                if total > 0:
                    if asset == 'USD':
                        print(f"  {asset}: ${total:,.2f} (Free: ${free:,.2f}, Locked: ${locked:,.2f})")
                    else:
                        print(f"  {asset}: {total:.6f} (Free: {free:.6f}, Locked: {locked:.6f})")
        
        # Get pending orders
        print("\n📋 Pending Orders:")
        pending = client.get_pending_count()
        
        if pending.get('Success'):
            total_pending = pending.get('TotalPending', 0)
            
            if total_pending > 0:
                print(f"  Total: {total_pending} orders")
                
                order_pairs = pending.get('OrderPairs', {})
                for pair, count in order_pairs.items():
                    print(f"    - {pair}: {count} order(s)")
            else:
                print("  No pending orders")
        
        # Get recent orders
        print("\n📈 Recent Orders (Last 10):")
        orders = client.query_order(limit=10)
        
        if orders.get('Success'):
            order_list = orders.get('OrderMatched', [])
            
            if order_list:
                for order in order_list:
                    order_id = order.get('OrderID')
                    pair = order.get('Pair')
                    side = order.get('Side')
                    status = order.get('Status')
                    qty = order.get('Quantity')
                    price = order.get('Price')
                    filled_qty = order.get('FilledQuantity', 0)
                    
                    status_emoji = {
                        'FILLED': '✅',
                        'PENDING': '⏳',
                        'CANCELED': '❌',
                        'PARTIALLY_FILLED': '🔄'
                    }.get(status, '❓')
                    
                    print(f"  {status_emoji} Order #{order_id}: {side} {qty:.6f} {pair} @ ${price:,.2f} [{status}]")
                    if filled_qty > 0 and filled_qty != qty:
                        print(f"      Filled: {filled_qty:.6f} / {qty:.6f}")
            else:
                print("  No recent orders")
        
        print("\n" + "=" * 60)
        print("✅ Status check complete")
        print("=" * 60)
    
    except RoostooAPIError as e:
        print(f"\n❌ API Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
