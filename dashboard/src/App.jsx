import React, { useState } from 'react';
import './index.css';

// Mocked output matching Python pipeline
const INITIAL_DATA = {
  regime: {
    mode: "HEDGE",
    adx_trend: 18.4,
    vol_target_scalar: 0.85,
    status: "Volatility Elevated - Limiting Exposure"
  },
  weights: [
    { ticker: "AAPL", currentWeight: "0%", targetWeight: "15%", delta: "+15%" },
    { ticker: "MSFT", currentWeight: "10%", targetWeight: "10%", delta: "0%" },
    { ticker: "NVDA", currentWeight: "12%", targetWeight: "0%", delta: "-12%" },
    { ticker: "TSLA", currentWeight: "0%", targetWeight: "5%", delta: "+5%" },
  ],
  alpacaStatus: "CONNECTED - PAPER"
};

function App() {
  const [data, setData] = useState(INITIAL_DATA);
  const [executing, setExecuting] = useState(false);
  const [executionLog, setExecutionLog] = useState([]);

  const changeMode = (mode) => {
    let status = mode === "HEDGE" ? "Volatility Elevated - Limiting Exposure" :
      mode === "BALANCED" ? "Momentum Tracked - Stable Allocation" : "Uncapped Exposure - Hunting Alpha";
    setData(prev => ({
      ...prev,
      regime: { ...prev.regime, mode, status, vol_target_scalar: mode === "HEDGE" ? 0.85 : mode === "BALANCED" ? 1.5 : 2.5 }
    }));
  };

  const handleExecute = () => {
    setExecuting(true);
    setExecutionLog(["Initiating Broker Protocol..."]);

    setTimeout(() => setExecutionLog(p => [...p, "> Fetching Live Account Balances from Alpaca..."]), 800);
    setTimeout(() => setExecutionLog(p => [...p, "> Calculating Differential Slivers..."]), 1600);
    setTimeout(() => setExecutionLog(p => [...p, "  ✓ Dispatched SELL NVDA (Liquidating $12,000)"]), 2400);
    setTimeout(() => setExecutionLog(p => [...p, "  ✓ Dispatched BUY AAPL (Deploying $15,000)"]), 3000);
    setTimeout(() => setExecutionLog(p => [...p, "  ✓ Dispatched BUY TSLA (Deploying $5,000)"]), 3600);
    setTimeout(() => {
      setExecutionLog(p => [...p, "SUCCESS: Live Portfolio Aligned with Target Weights."]);
      setExecuting(false);
      setData(prev => ({
        ...prev,
        weights: prev.weights.map(w => ({ ...w, currentWeight: w.targetWeight, delta: "0%" }))
      }));
    }, 4500);
  };

  return (
    <div className="dashboard-container">
      <header className="flex-row animate-fade-in" style={{ justifyContent: 'space-between', marginBottom: '2.5rem' }}>
        <div className="flex-col" style={{ gap: '0.2rem' }}>
          <h1 className="text-cyan">HailMary Quantitative Engine</h1>
          <p className="text-muted">Live Machine Learning Signal Architecture</p>
        </div>
        <div className="glass-panel flex-row" style={{ padding: '0.8rem 1.5rem', borderRadius: '30px' }}>
          <div className="pulse-orb"></div>
          <span style={{ fontWeight: 600, letterSpacing: '0.05em' }}>{data.alpacaStatus}</span>
        </div>
      </header>

      <div className="grid-3 animate-fade-in delay-1">
        {/* Risk Profile Selector */}
        <div className="glass-panel flex-col">
          <h3 className="text-muted">ACTIVE RISK PROFILE</h3>
          <div className="flex-col" style={{ marginTop: '0.5rem' }}>
            {['HEDGE', 'BALANCED', 'GROWTH'].map(m => (
              <button
                key={m}
                onClick={() => changeMode(m)}
                className={`badge ${data.regime.mode === m ? 'active' : ''}`}
                style={{ width: '100%', textAlign: 'left', cursor: 'pointer', display: 'flex', justifyContent: 'space-between' }}
              >
                <span>{m} MODE</span>
                {data.regime.mode === m && <span className="text-cyan">⚙</span>}
              </button>
            ))}
          </div>
        </div>

        {/* Global Regime State */}
        <div className="glass-panel flex-col" style={{ gridColumn: 'span 2' }}>
          <div className="flex-row" style={{ justifyContent: 'space-between' }}>
            <h3 className="text-muted">MARKET REGIME FILTER</h3>
            <span className={data.regime.vol_target_scalar >= 1.5 ? "text-pos" : "text-neg"} style={{ fontWeight: 700 }}>
              {data.regime.status}
            </span>
          </div>

          <div className="grid-2" style={{ marginTop: '1rem' }}>
            <div className="glass-panel" style={{ background: 'rgba(0,0,0,0.3)', border: 'none' }}>
              <p className="text-muted">ADX STR. (TREND)</p>
              <h2 className="text-cyan" style={{ fontSize: '2.5rem' }}>{data.regime.adx_trend}</h2>
            </div>
            <div className="glass-panel" style={{ background: 'rgba(0,0,0,0.3)', border: 'none' }}>
              <p className="text-muted">LEVERAGE MULTIPLIER</p>
              <h2 className="text-purple" style={{ fontSize: '2.5rem' }}>{data.regime.vol_target_scalar}x</h2>
            </div>
          </div>
        </div>
      </div>

      <div className="grid-2 animate-fade-in delay-2" style={{ marginTop: '2rem' }}>

        {/* ML Outputs Panel */}
        <div className="glass-panel flex-col">
          <div className="flex-row" style={{ justifyContent: 'space-between' }}>
            <h3>TARGET PORTFOLIO WEIGHTS</h3>
            <span className="badge">LGBM + RF BAGGED</span>
          </div>

          <table>
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Current</th>
                <th>Target</th>
                <th>Delta</th>
              </tr>
            </thead>
            <tbody>
              {data.weights.map((row, i) => (
                <tr key={i}>
                  <td style={{ fontWeight: 600 }} className="text-cyan">{row.ticker}</td>
                  <td className="text-muted">{row.currentWeight}</td>
                  <td>{row.targetWeight}</td>
                  <td className={row.delta.startsWith('+') ? "text-pos" : row.delta === "0%" ? "text-muted" : "text-neg"}>
                    {row.delta}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Execution Terminal */}
        <div className="glass-panel flex-col" style={{ justifyContent: 'space-between' }}>
          <div className="flex-col">
            <div className="flex-row" style={{ justifyContent: 'space-between' }}>
              <h3>LIVE EXECUTION TERMINAL</h3>
              <span className="badge">ROUTING: ALPACA</span>
            </div>

            <div style={{
              background: '#040609',
              borderRadius: '8px',
              padding: '1rem',
              marginTop: '1rem',
              height: '180px',
              fontFamily: 'monospace',
              fontSize: '0.85rem',
              color: '#00E5FF',
              overflowY: 'auto'
            }}>
              {executionLog.length === 0 ? (
                <span style={{ color: '#444' }}>Awaiting execution command...</span>
              ) : (
                executionLog.map((log, i) => <div key={i} style={{ marginBottom: '0.5rem' }}>{log}</div>)
              )}
            </div>
          </div>

          <button
            className="btn-primary"
            style={{ marginTop: '1rem', position: 'relative', overflow: 'hidden' }}
            onClick={handleExecute}
            disabled={executing}
          >
            {executing ? (
              <>
                <div className="pulse-orb" style={{ background: 'white', border: 'none', boxShadow: 'none' }}></div>
                TRANSMITTING TO BROKER...
              </>
            ) : "EXECUTE PORTFOLIO ROTATION"}
          </button>
        </div>

      </div>
    </div>
  );
}

export default App;
