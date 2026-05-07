"""
Live Execution Orchestrator -> Translates ML Signal Weights into Live Broker REST APIs.
"""
import pandas as pd
from typing import Dict, Any

class BaseBrokerExecution:
    """
    Abstract blueprint for executing SOTA Quant ML signals on a live broker.
    Handles calculating target portfolio differentials and pushing market API orders.
    """
    def __init__(self, api_keys: Dict[str, str], is_paper: bool = True):
        self.keys = api_keys
        self.is_paper = is_paper
        
    def get_live_portfolio(self) -> Dict[str, float]:
        """Fetch current holdings dollar-value from broker"""
        raise NotImplementedError
        
    def get_total_equity(self) -> float:
        """Fetch net liquidity/equity"""
        raise NotImplementedError
        
    def submit_market_order(self, symbol: str, dollar_amount: float, side: str):
        """Dispatches actual REST API request to broker"""
        raise NotImplementedError
        
    def execute_target_weights(self, target_weights: pd.Series):
        """
        The core bridging algorithm.
        1. Reads ML Target Weights Array
        2. Compares to Live Server Portfolio
        3. Fires differential rebalancing orders
        """
        equity = self.get_total_equity()
        current_holdings = self.get_live_portfolio()
        
        target_dollars = target_weights * equity
        print(f"Executing Multi-Leg Rebalance on Portfolio Equity: ${equity:,.2f}")
        
        # 1. Sell-Side (Liquidate decreasing positions first to free cash for buys)
        for symbol, current_val in current_holdings.items():
            target_val = target_dollars.get(symbol, 0.0)
            if target_val < current_val:
                sell_amount = current_val - target_val
                # Apply 0.5% haircut: Alpaca computes share qty from notional at current
                # price, which can exceed available shares if price ticked up since fetch.
                sell_amount = sell_amount * 0.995
                if sell_amount > (equity * 0.005): # Min threshold 0.5% turnover
                    self.submit_market_order(symbol, sell_amount, 'SELL')
                    
        # 2. Buy-Side (Allocate cash to upgrading positions)
        for symbol, target_val in target_dollars.items():
            current_val = current_holdings.get(symbol, 0.0)
            if target_val > current_val:
                buy_amount = target_val - current_val
                if buy_amount > (equity * 0.005):
                    self.submit_market_order(symbol, buy_amount, 'BUY')

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class AlpacaLiveExecution(BaseBrokerExecution):
    """Implementation for US Equities (Alpaca Broker API)"""
    
    def __init__(self, api_keys: Dict[str, str], is_paper: bool = True):
        super().__init__(api_keys, is_paper)
        self.client = TradingClient(self.keys['API_KEY'], self.keys['SECRET_KEY'], paper=self.is_paper)
    
    def get_live_portfolio(self) -> Dict[str, float]:
        positions = self.client.get_all_positions()
        return {p.symbol: float(p.market_value) for p in positions}
        
    def get_total_equity(self) -> float:
        return float(self.client.get_account().equity)
        
    def submit_market_order(self, symbol: str, dollar_amount: float, side: str):
        print(f"[ALPACA] -> Dispatched {side} ${dollar_amount:,.2f} of {symbol}")
        try:
            req = MarketOrderRequest(
                symbol=symbol,
                notional=round(dollar_amount, 2),
                side=OrderSide.BUY if side == 'BUY' else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            self.client.submit_order(order_data=req)
            print(f"   [OK] Order Transmitted to Server")
        except Exception as e:
            print(f"   X Failed to transmit order: {e}")

class ZerodhaIndianExecution(BaseBrokerExecution):
    """Implementation for Indian Equities (Zerodha/Kite API)"""
    
    def get_live_portfolio(self) -> Dict[str, float]:
        return {'RELIANCE': 50000.0} 
        
    def get_total_equity(self) -> float:
        return 1000000.0
        
    def submit_market_order(self, symbol: str, dollar_amount: float, side: str):
        print(f"[ZERODHA] -> Dispatched {side} ₹{dollar_amount:,.2f} of {symbol}")

if __name__ == '__main__':
    # Test execution simulation
    ml_output = pd.Series({'AAPL': 0.35, 'NVDA': 0.15, 'MSFT': 0.00})
    broker = AlpacaLiveExecution(api_keys={}, is_paper=True)
    broker.execute_target_weights(ml_output)
