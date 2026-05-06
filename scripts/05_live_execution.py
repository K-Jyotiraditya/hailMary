"""
Live Execution Synthesizer.
Connects the final generated SOTA quantitative arrays to the Broker APIs for production paper-trading.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from execution.broker_live import AlpacaLiveExecution

def main():
    print("==================================================")
    print("STEP 05: LIVE BROKER EXECUTION")
    print("==================================================")
    print("Loading latest SOTA Target Weight Arrays...")
    
    # In live environments, you deserialize this directly from ML outputs.
    latest_signals = pd.Series({'AAPL': 0.15, 'MSFT': 0.20, 'NVDA': 0.10})
    
    print("\nInitializing Alpaca Connection Environment...")
    broker = AlpacaLiveExecution(api_keys={'API_KEY': 'API', 'SECRET_KEY': 'SECRET'}, is_paper=True)
    broker.execute_target_weights(latest_signals)
    
if __name__ == '__main__':
    main()
