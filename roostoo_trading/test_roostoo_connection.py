"""
Test Roostoo API connection and credentials
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roostoo_trading import RoostooClient, RoostooAPIError
from roostoo_trading.config import ROOSTOO_CONFIG


def test_connection():
    """Test Roostoo API connection"""
    
    print("=" * 60)
    print("Testing Roostoo API Connection")
    print("=" * 60)
    print()
    
    # Get credentials
    api_key = ROOSTOO_CONFIG.get('api_key')
    secret_key = ROOSTOO_CONFIG.get('secret_key')
    base_url = ROOSTOO_CONFIG.get('base_url')
    
    if not api_key or not secret_key:
        print("❌ Error: API credentials not found in roostoo_config.py")
        return False
    
    print(f"API URL: {base_url}")
    print(f"API Key: {api_key[:20]}...")
    print()
    
    # Initialize client
    try:
        client = RoostooClient(
            api_key=api_key,
            secret_key=secret_key,
            base_url=base_url
        )
        print("✅ Client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return False
    
    # Test 1: Get balance
    print("\n" + "-" * 60)
    print("Test 1: Get Account Balance")
    print("-" * 60)
    
    try:
        balance = client.get_balance()
        
        if balance.get('Success'):
            print("✅ Balance retrieved successfully")
            
            wallet = balance.get('Wallet', {})
            print("\nAccount Balances:")
            
            for asset, amounts in wallet.items():
                free = amounts.get('Free', 0)
                locked = amounts.get('Lock', 0)
                total = free + locked
                
                if total > 0:
                    if asset == 'USD':
                        print(f"  {asset}: ${total:,.2f} (Free: ${free:,.2f}, Locked: ${locked:,.2f})")
                    else:
                        print(f"  {asset}: {total:.6f} (Free: {free:.6f}, Locked: {locked:.6f})")
        else:
            print(f"❌ Failed: {balance.get('ErrMsg')}")
            return False
    
    except RoostooAPIError as e:
        print(f"❌ API Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    # Test 2: Get pending orders
    print("\n" + "-" * 60)
    print("Test 2: Get Pending Orders")
    print("-" * 60)
    
    try:
        pending = client.get_pending_count()
        
        if pending.get('Success'):
            print("✅ Pending orders retrieved successfully")
            
            total = pending.get('TotalPending', 0)
            print(f"\nTotal pending orders: {total}")
            
            if total > 0:
                pairs = pending.get('OrderPairs', {})
                for pair, count in pairs.items():
                    print(f"  {pair}: {count} order(s)")
        else:
            # It's OK if there are no pending orders
            if "no pending order" in pending.get('ErrMsg', '').lower():
                print("✅ No pending orders (this is OK)")
            else:
                print(f"❌ Failed: {pending.get('ErrMsg')}")
                return False
    
    except RoostooAPIError as e:
        print(f"❌ API Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    # Test 3: Query recent orders
    print("\n" + "-" * 60)
    print("Test 3: Query Recent Orders")
    print("-" * 60)
    
    try:
        orders = client.query_order(limit=5)
        
        if orders.get('Success'):
            print("✅ Orders retrieved successfully")
            
            order_list = orders.get('OrderMatched', [])
            
            if order_list:
                print(f"\nFound {len(order_list)} recent orders:")
                for order in order_list[:3]:
                    order_id = order.get('OrderID')
                    pair = order.get('Pair')
                    side = order.get('Side')
                    status = order.get('Status')
                    qty = order.get('Quantity')
                    price = order.get('Price')
                    
                    print(f"  Order #{order_id}: {side} {qty:.6f} {pair} @ ${price:,.2f} [{status}]")
            else:
                print("\nNo order history yet (this is OK for new accounts)")
        else:
            # It's OK if there are no orders
            if "no order matched" in orders.get('ErrMsg', '').lower():
                print("✅ No order history (this is OK for new accounts)")
            else:
                print(f"❌ Failed: {orders.get('ErrMsg')}")
                return False
    
    except RoostooAPIError as e:
        print(f"❌ API Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    # All tests passed
    print("\n" + "=" * 60)
    print("✅ All tests passed! Your Roostoo API is working correctly.")
    print("=" * 60)
    print()
    print("You can now run the live trading bot:")
    print("  python live_vcre_trader.py --dry-run  # Test mode")
    print("  python live_vcre_trader.py            # Live trading")
    print()
    
    return True


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
