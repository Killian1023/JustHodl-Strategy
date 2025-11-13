#!/usr/bin/env python3
"""
Emergency script to fully clear account:
1. Cancel all pending orders
2. Close all open positions
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roostoo_trading import RoostooClient
from roostoo_trading.config import ROOSTOO_CONFIG

# Try to import from alma config, fallback to vcre
try:
    from live_alma_config import QUANTITY_STEP_SIZES, PRICE_STEP_SIZES
except ImportError:
    try:
        from utils.live_trading_config import QUANTITY_STEP_SIZES, PRICE_STEP_SIZES
    except ImportError:
        # Fallback defaults
        QUANTITY_STEP_SIZES = {}
        PRICE_STEP_SIZES = {}


def round_quantity(symbol: str, quantity: float) -> float:
    """Round quantity to proper step size"""
    step_size = QUANTITY_STEP_SIZES.get(symbol, 0.001)
    
    if step_size >= 1:
        decimal_places = 0
    else:
        decimal_places = len(str(step_size).split('.')[-1])
    
    scale = 10 ** decimal_places
    rounded = int(quantity * scale / (step_size * scale)) * step_size
    rounded = round(rounded, decimal_places)
    
    return rounded


def cancel_all_orders(client):
    """Cancel all pending orders"""
    print("\n" + "=" * 60)
    print("STEP 1: Cancelling All Pending Orders")
    print("=" * 60)
    
    # Get pending orders count
    print("\n→ Checking pending orders...")
    try:
        pending_response = client.get_pending_count()
        if not pending_response.get('Success'):
            print(f"❌ Failed to get pending count: {pending_response.get('ErrMsg')}")
            return False
        
        pending_count = pending_response.get('PendingCount', 0)
        print(f"✅ Pending count API reports: {pending_count}")
        
    except Exception as e:
        print(f"❌ Error getting pending count: {e}")
        return False
    
    # Get order details - try multiple methods
    print("\n→ Fetching order details...")
    orders = []
    
    try:
        # Method 1: Query with pending_only flag
        orders_response = client.query_order(pending_only=True)
        if orders_response.get('Success'):
            orders = orders_response.get('OrderMatched', []) or orders_response.get('OrderList', [])
            print(f"✅ Method 1: Retrieved {len(orders)} order(s)")
        
        # Method 2: If no orders found, try querying all recent orders
        if len(orders) == 0:
            print("→ Trying alternative method...")
            orders_response = client.query_order(pending_only=False)
            if orders_response.get('Success'):
                all_orders = orders_response.get('OrderMatched', []) or orders_response.get('OrderList', [])
                # Filter for pending/partially filled orders
                orders = [o for o in all_orders if o.get('Status') in ['Pending', 'PartiallyFilled', 'Open']]
                print(f"✅ Method 2: Found {len(orders)} pending order(s) from {len(all_orders)} total")
        
    except Exception as e:
        print(f"❌ Error getting orders: {e}")
        return False
    
    if len(orders) == 0:
        print("✅ No pending orders found to cancel")
        return True
    
    # Display orders
    print("\n→ Pending Orders:")
    print("-" * 60)
    for i, order in enumerate(orders, 1):
        order_id = order.get('OrderID')
        pair = order.get('Pair')
        side = order.get('Side')
        quantity = order.get('Quantity', 0)
        price = order.get('Price', 0)
        
        print(f"  #{i}: {pair} {side} {quantity:.4f} @ ${price:.6f} (ID: {order_id})")
    print("-" * 60)
    
    # Cancel orders
    print("\n→ Cancelling orders...")
    cancelled_count = 0
    failed_count = 0
    
    for order in orders:
        order_id = order.get('OrderID')
        pair = order.get('Pair')
        
        try:
            response = client.cancel_order(order_id)
            
            if response.get('Success'):
                print(f"  ✅ Cancelled Order {order_id} ({pair})")
                cancelled_count += 1
            else:
                error_msg = response.get('ErrMsg', 'Unknown error')
                print(f"  ❌ Failed Order {order_id}: {error_msg}")
                failed_count += 1
            
            time.sleep(0.2)  # Small delay between cancellations
                
        except Exception as e:
            print(f"  ❌ Error Order {order_id}: {e}")
            failed_count += 1
    
    print(f"\n✅ Cancelled: {cancelled_count} | ❌ Failed: {failed_count}")
    return failed_count == 0


def close_all_positions(client: RoostooClient, exclude_assets: list = None) -> bool:
    """Close all open positions using market orders"""
    print("\n" + "=" * 60)
    print("STEP 2: Closing All Open Positions")
    print("=" * 60)
    
    # Get current balance
    print("\n→ Checking current holdings...")
    try:
        balance_response = client.get_balance()
        if not balance_response.get('Success'):
            print(f"❌ Failed to get balance: {balance_response.get('ErrMsg')}")
            return False
        
        wallet = balance_response.get('SpotWallet', balance_response.get('Wallet', {}))
        print(f"✅ Balance retrieved ({len(wallet)} assets)")
        
    except Exception as e:
        print(f"❌ Error getting balance: {e}")
        return False
    
    # Find all non-USD holdings (excluding specified assets)
    if exclude_assets is None:
        exclude_assets = []
    
    positions_to_close = []
    for asset, balances in wallet.items():
        if asset == 'USD':
            continue
        
        # Skip excluded assets (e.g., BTC for ALMA since LSTM trades it)
        if asset in exclude_assets:
            print(f"  ⏭️  Skipping {asset} (reserved for other strategy)")
            continue
        
        free = balances.get('Free', 0)
        locked = balances.get('Lock', 0)
        total = free + locked
        
        if total > 0:
            positions_to_close.append({
                'asset': asset,
                'free': free,
                'locked': locked,
                'total': total
            })
    
    if not positions_to_close:
        print("✅ No positions to close")
        return True
    
    # Display positions
    print(f"\n→ Found {len(positions_to_close)} position(s):")
    print("-" * 60)
    for pos in positions_to_close:
        print(f"  {pos['asset']}: {pos['total']:.4f} (Free: {pos['free']:.4f}, Locked: {pos['locked']:.4f})")
    print("-" * 60)
    
    # Close each position
    print("\n→ Closing positions...")
    closed_count = 0
    failed_count = 0
    skipped_count = 0
    
    for pos in positions_to_close:
        asset = pos['asset']
        quantity = pos['free']
        
        if quantity <= 0:
            print(f"  ⚠️  {asset}: Skipped (no free balance, {pos['locked']:.4f} locked)")
            skipped_count += 1
            continue
        
        symbol = f"{asset}/USD"
        rounded_qty = round_quantity(symbol, quantity)
        
        if rounded_qty <= 0:
            print(f"  ⚠️  {asset}: Quantity too small ({quantity:.6f} -> {rounded_qty:.6f})")
            skipped_count += 1
            continue
        
        try:
            response = client.sell(symbol, rounded_qty)
            
            if response.get('Success'):
                order_detail = response.get('OrderDetail', {})
                filled_price = order_detail.get('FilledAverPrice', 0)
                order_id = order_detail.get('OrderID')
                print(f"  ✅ {symbol}: Sold {rounded_qty:.4f} @ ${filled_price:.4f} (ID: {order_id})")
                closed_count += 1
            else:
                error_msg = response.get('ErrMsg', 'Unknown error')
                print(f"  ❌ {symbol}: {error_msg}")
                failed_count += 1
            
            time.sleep(0.2)  # Small delay between orders
                
        except Exception as e:
            print(f"  ❌ {symbol}: {e}")
            failed_count += 1
    
    print(f"\n✅ Closed: {closed_count} | ⚠️  Skipped: {skipped_count} | ❌ Failed: {failed_count}")
    return failed_count == 0


def clear_account():
    """Main function to clear entire account"""
    print("=" * 60)
    print("🚨 EMERGENCY ACCOUNT CLEARER 🚨")
    print("=" * 60)
    print("\nThis will:")
    print("  1. Cancel ALL pending orders")
    print("  2. Close ALL open positions (MARKET orders)")
    print("\n⚠️  WARNING: This action CANNOT be undone!")
    print("=" * 60)
    
    confirm = input("\nType 'CLEAR' (all caps) to proceed: ")
    
    if confirm != 'CLEAR':
        print("\n❌ Cancelled - Account NOT cleared")
        return False
    
    # Get credentials
    api_key = os.getenv('ROOSTOO_API_KEY') or ROOSTOO_CONFIG.get('api_key')
    secret_key = os.getenv('ROOSTOO_SECRET_KEY') or ROOSTOO_CONFIG.get('secret_key')
    
    if not api_key or not secret_key:
        print("\n❌ ERROR: Roostoo credentials not found")
        return False
    
    # Create client
    client = RoostooClient(api_key, secret_key)
    
    # Step 1: Cancel all orders
    orders_success = cancel_all_orders(client)
    
    # Wait a moment for cancellations to process
    if orders_success:
        print("\n⏳ Waiting 2 seconds for cancellations to process...")
        time.sleep(2)
    
    # Step 2: Close all positions
    positions_success = close_all_positions(client)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    if orders_success and positions_success:
        print("✅ Account successfully cleared!")
        print("   - All orders cancelled")
        print("   - All positions closed")
    elif orders_success:
        print("⚠️  Partial success:")
        print("   ✅ All orders cancelled")
        print("   ❌ Some positions failed to close")
    elif positions_success:
        print("⚠️  Partial success:")
        print("   ❌ Some orders failed to cancel")
        print("   ✅ All positions closed")
    else:
        print("❌ Failed to fully clear account")
        print("   Please check errors above and try again")
    
    print("=" * 60)
    
    return orders_success and positions_success


if __name__ == "__main__":
    success = clear_account()
    sys.exit(0 if success else 1)
