"""
Backtesting Module for LSTM Price Direction Strategy

Simulates trading based on model predictions and calculates performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt


class TradingBacktester:
    """
    Backtester for price direction prediction strategy.
    
    Simulates trading based on model predictions and calculates:
    - Total return
    - Sharpe ratio
    - Maximum drawdown
    - Win rate
    - Profit factor
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.001,  # 0.1% commission
        position_size: float = 1.0   # Fraction of capital to use per trade
    ):
        """
        Args:
            initial_capital: Starting capital
            commission: Trading commission (as fraction)
            position_size: Position size as fraction of capital
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.position_size = position_size
        
        self.trades = []
        self.equity_curve = []
        self.results = {}
    
    def run_backtest(
        self,
        prices: np.ndarray,
        predictions: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run backtest simulation.
        
        Args:
            prices: Array of prices
            predictions: Array of predictions (1 for up, 0 for down)
            timestamps: Optional timestamps
            
        Returns:
            Dictionary of backtest results
        """
        prices = prices.flatten()
        predictions = predictions.flatten()
        
        if len(prices) != len(predictions):
            raise ValueError("Prices and predictions must have same length")
        
        capital = self.initial_capital
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0
        
        self.trades = []
        self.equity_curve = [capital]
        
        for i in range(len(predictions) - 1):
            current_price = prices[i]
            next_price = prices[i + 1]
            prediction = predictions[i]
            
            # Close existing position
            if position != 0:
                # Calculate P&L
                if position == 1:  # Long position
                    pnl = (next_price - entry_price) / entry_price
                else:  # Short position
                    pnl = (entry_price - next_price) / entry_price
                
                # Apply commission
                pnl -= self.commission * 2  # Entry and exit
                
                # Update capital
                trade_size = capital * self.position_size
                capital += trade_size * pnl
                
                # Record trade
                self.trades.append({
                    'entry_price': entry_price,
                    'exit_price': next_price,
                    'position': 'long' if position == 1 else 'short',
                    'pnl': pnl,
                    'pnl_amount': trade_size * pnl,
                    'timestamp': timestamps[i] if timestamps is not None else i
                })
                
                position = 0
            
            # Open new position based on prediction
            if prediction == 1:  # Predict up -> go long
                position = 1
                entry_price = next_price
            elif prediction == 0:  # Predict down -> go short
                position = -1
                entry_price = next_price
            
            self.equity_curve.append(capital)
        
        # Calculate performance metrics
        self.results = self._calculate_metrics()
        
        return self.results
    
    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.trades:
            return {
                'total_return': 0.0,
                'total_return_pct': 0.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0
            }
        
        # Total return
        final_capital = self.equity_curve[-1]
        total_return = final_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Trade statistics
        num_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(t['pnl_amount'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl_amount'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe ratio (annualized, assuming daily data)
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        sharpe_ratio = np.sqrt(252) * (np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0
        
        # Maximum drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = equity_array - running_max
        max_drawdown = np.min(drawdown)
        max_drawdown_pct = (max_drawdown / np.max(running_max)) * 100 if np.max(running_max) > 0 else 0
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'num_trades': num_trades,
            'num_winning_trades': len(winning_trades),
            'num_losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'avg_win': np.mean([t['pnl_amount'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t['pnl_amount'] for t in losing_trades]) if losing_trades else 0,
            'final_capital': final_capital
        }
    
    def print_results(self):
        """Print backtest results."""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        print(f"\nInitial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Capital: ${self.results['final_capital']:,.2f}")
        print(f"Total Return: ${self.results['total_return']:,.2f} ({self.results['total_return_pct']:.2f}%)")
        
        print("\nTrade Statistics:")
        print("-"*60)
        print(f"Total Trades: {self.results['num_trades']}")
        print(f"Winning Trades: {self.results['num_winning_trades']}")
        print(f"Losing Trades: {self.results['num_losing_trades']}")
        print(f"Win Rate: {self.results['win_rate']*100:.2f}%")
        print(f"Profit Factor: {self.results['profit_factor']:.2f}")
        
        print("\nRisk Metrics:")
        print("-"*60)
        print(f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: ${self.results['max_drawdown']:,.2f} ({self.results['max_drawdown_pct']:.2f}%)")
        
        print("\nAverage Trade:")
        print("-"*60)
        print(f"Average Win: ${self.results['avg_win']:,.2f}")
        print(f"Average Loss: ${self.results['avg_loss']:,.2f}")
        
        print("="*60 + "\n")
    
    def plot_equity_curve(self, save_path: Optional[str] = None):
        """
        Plot equity curve.
        
        Args:
            save_path: Path to save figure (optional)
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve, linewidth=2)
        plt.axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
        plt.title('Equity Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Time Period')
        plt.ylabel('Capital ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Equity curve saved to {save_path}")
        
        plt.show()
    
    def plot_drawdown(self, save_path: Optional[str] = None):
        """
        Plot drawdown chart.
        
        Args:
            save_path: Path to save figure (optional)
        """
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = equity_array - running_max
        drawdown_pct = (drawdown / running_max) * 100
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(range(len(drawdown_pct)), drawdown_pct, 0, alpha=0.3, color='red')
        plt.plot(drawdown_pct, color='red', linewidth=2)
        plt.title('Drawdown', fontsize=14, fontweight='bold')
        plt.xlabel('Time Period')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Drawdown chart saved to {save_path}")
        
        plt.show()
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """
        Get trades as DataFrame.
        
        Returns:
            DataFrame with trade details
        """
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)
    
    def calculate_monthly_returns(self, timestamps: pd.DatetimeIndex) -> pd.Series:
        """
        Calculate monthly returns.
        
        Args:
            timestamps: DatetimeIndex for equity curve
            
        Returns:
            Series of monthly returns
        """
        if len(timestamps) != len(self.equity_curve):
            raise ValueError("Timestamps length must match equity curve length")
        
        equity_df = pd.DataFrame({
            'timestamp': timestamps,
            'equity': self.equity_curve
        })
        equity_df.set_index('timestamp', inplace=True)
        
        monthly_equity = equity_df.resample('M').last()
        monthly_returns = monthly_equity.pct_change() * 100
        
        return monthly_returns['equity']
