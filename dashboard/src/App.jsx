import React, { useState, useEffect } from 'react';
import './index.css';

// Synthetic structural data mirroring live algorithm state
const SYS_STATE = {
  modelHealth: {
    backtestIC: 0.0434, // S&P 100 actual extracted IC
    liveIC: 0.0392,
    featureDrift: 0.012, // PSI index
    nextRetrain: "14 Days",
    pbo: 0.91,
    dsr: 1.84,
    baggingConf: "±1.22%"
  },
  regime: {
    adxScore: 18.4,
    adxLabel: "Weak Trend → Limiting Base Exposure",
    sma200Dist: "-1.5%",
    combinedScore: 0.35,
    realizedVol: 0.142,
    targetVol: 0.12,
    scaler: 0.84, // 12/14.2
    stripHistory: Array.from({ length: 60 }, (_, i) => Math.random() > 0.8 ? "danger" : Math.random() > 0.4 ? "warning" : "safe")
  },
  performance: {
    maxDD: "-3.41%",
    currentDD: "-0.84%",
    hitRate: "54.6%",
    sharpe60d: 1.12,
    mtd: "+1.4%",
    ytd: "+8.2%"
  },
  positions: [
    { ticker: "AAPL", exp: "15%", attr: "vol_60(-1.2) mom_20(+2.1)", drift: 0.01 },
    { ticker: "MSFT", exp: "10%", attr: "mean_abs_ic(+1.5) MACD(+0.8)", drift: 0.02 },
    { ticker: "NVDA", exp: "12%", attr: "dollar_vol(+2.4) vs_vwap(+1.1)", drift: -0.01 }
  ],
  execution: {
    routing: "ALPACA (US EQ)",
    cash: "$63,000",
    nav: "$100,000",
    slippageModel: "ADV Square-root (Est. 2bps)",
    lastFetch: "2026-05-06T15:55:01Z"
  }
};

function App() {
  const [executing, setExecuting] = useState(false);
  const [log, setLog] = useState([]);

  const handleRebalance = () => {
    setExecuting(true);
    setLog(["[SYS] Dispatching API Call to Execution Node..."]);
    setTimeout(() => setLog(p => [...p, "[API] NAV: 100000.0 | Cash: 63000.0"]), 500);
    setTimeout(() => setLog(p => [...p, "[EXEC] AAPL Target Diff: $15,000 -> Constructing Bracket..."]), 1200);
    setTimeout(() => setLog(p => [...p, "[EXEC] NVDA Target Diff: -$12,000 -> Constructing MKT..."]), 1600);
    setTimeout(() => setLog(p => [...p, "[FILL] NVDA 124 shares @ MKT | Realized Slippage: 1.4bps"]), 2200);
    setTimeout(() => setLog(p => [...p, "[FILL] AAPL 88 shares @ MKT | Realized Slippage: 1.8bps"]), 2600);
    setTimeout(() => {
      setLog(p => [...p, "[SYS] Recon Complete. Target weights secured."]);
      setExecuting(false);
    }, 3200);
  };

  return (
    <div className="dashboard-container">
      <header className="terminal-header">
        <div>
          <h1 className="terminal-title">HAILMARY OEX / S&P 100</h1>
          <span className="text-dim">SYS_VER: P5 COMBINATORIAL PURGED | OOS DEFLATED (DSR: {SYS_STATE.modelHealth.dsr})</span>
        </div>
        <div style={{ textAlign: 'right' }}>
          <span className="text-info tracking-wide">LIVE.ALPACA</span><br />
          <span className="text-dim">{SYS_STATE.execution.lastFetch}</span>
        </div>
      </header>

      <div className="grid-main">
        {/* LEFT COLUMN: PERFORMANCE & HEALTH */}
        <div className="flex-col">
          <div className="panel">
            <div className="panel-header">Performance & Risk</div>
            <div className="grid-2">
              <div>
                <div className="metric-label">CURRENT DD vs MAX DD</div>
                <div className="metric-val text-white">{SYS_STATE.performance.currentDD} / <span className="text-dim">{SYS_STATE.performance.maxDD}</span></div>
              </div>
              <div>
                <div className="metric-label">ROLLING 60D SHARPE</div>
                <div className="metric-val text-info">{SYS_STATE.performance.sharpe60d}</div>
              </div>
              <div>
                <div className="metric-label">HIT RATE (WIN/LOSS)</div>
                <div className="metric-val text-white">{SYS_STATE.performance.hitRate}</div>
              </div>
              <div>
                <div className="metric-label">MTD / YTD RET</div>
                <div className="metric-val text-pos">{SYS_STATE.performance.mtd} / {SYS_STATE.performance.ytd}</div>
              </div>
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">Model Health & Validation</div>
            <div className="flex-between">
              <span className="text-dim">LIVE IC VS BACKTEST IC</span>
              <span className="text-white">{SYS_STATE.modelHealth.liveIC} / {SYS_STATE.modelHealth.backtestIC}</span>
            </div>
            <div className="progress-bar-bg"><div className="progress-bar-fill" style={{ width: `${(SYS_STATE.modelHealth.liveIC / SYS_STATE.modelHealth.backtestIC) * 100}%`, background: 'var(--positive)' }}></div></div>

            <div className="flex-between" style={{ marginTop: '0.4rem' }}>
              <span className="text-dim">FEATURE DRIFT (PSI)</span>
              <span className={SYS_STATE.modelHealth.featureDrift > 0.05 ? "text-neg" : "text-white"}>{SYS_STATE.modelHealth.featureDrift}</span>
            </div>
            <div className="flex-between">
              <span className="text-dim">PROBABILITY OF OVERFIT (PBO)</span>
              <span className="text-white">{SYS_STATE.modelHealth.pbo}</span>
            </div>
            <div className="flex-between">
              <span className="text-dim">BAGGED INTERVAL</span>
              <span className="text-dim">{SYS_STATE.modelHealth.baggingConf}</span>
            </div>
          </div>
        </div>

        {/* MIDDLE COLUMN: REGIME & POSITIONS */}
        <div className="flex-col">
          <div className="panel">
            <div className="panel-header flex-between">
              <span>Composite Market Regime</span>
              <span>VOL TARGET: {SYS_STATE.regime.targetVol * 100}%</span>
            </div>
            <div className="grid-3" style={{ marginBottom: '1rem' }}>
              <div>
                <div className="metric-label">SMA-200 DISTANCE</div>
                <div className="metric-val text-neg">{SYS_STATE.regime.sma200Dist}</div>
              </div>
              <div>
                <div className="metric-label">ADX STR (TREND)</div>
                <div className="metric-val text-dim">{SYS_STATE.regime.adxScore}</div>
                <div style={{ fontSize: '0.7rem', color: '#666' }}>{SYS_STATE.regime.adxLabel}</div>
              </div>
              <div>
                <div className="metric-label">COMPOSITE MULTIPLIER</div>
                <div className="metric-val text-info">{(SYS_STATE.regime.scaler * SYS_STATE.regime.combinedScore).toFixed(2)}x</div>
              </div>
            </div>

            <div className="metric-label">TRAILING 60-DAY REGIME STATE TRANSITIONS</div>
            <div className="regime-strip">
              {SYS_STATE.regime.stripHistory.map((s, i) => (
                <div key={i} className={`strip-block strip-${s}`}></div>
              ))}
            </div>
          </div>

          <div className="panel" style={{ flex: 1 }}>
            <div className="panel-header">Active Alignments & Feature Attribution</div>
            <table>
              <thead>
                <tr>
                  <th>Component</th>
                  <th>Target Exp</th>
                  <th>Feature Drivers (Attribution)</th>
                  <th>Drift Z-Score</th>
                </tr>
              </thead>
              <tbody>
                {SYS_STATE.positions.map((p, i) => (
                  <tr key={i}>
                    <td className="text-white font-weight-700">{p.ticker}</td>
                    <td className="text-info">{p.exp}</td>
                    <td className="text-dim">{p.attr}</td>
                    <td className={p.drift > 0 ? "text-pos" : "text-neg"}>{p.drift}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* RIGHT COLUMN: OP EXECUTION */}
        <div className="flex-col">
          <div className="panel">
            <div className="panel-header">Execution Subsystem</div>
            <div className="flex-between">
              <span className="text-dim">ROUTING</span>
              <span className="text-white">{SYS_STATE.execution.routing}</span>
            </div>
            <div className="flex-between">
              <span className="text-dim">SLIPPAGE MODEL</span>
              <span className="text-dim">{SYS_STATE.execution.slippageModel}</span>
            </div>
            <div className="flex-between">
              <span className="text-dim">MARGIN AVAIL / NAV</span>
              <span className="text-white">{SYS_STATE.execution.cash} / {SYS_STATE.execution.nav}</span>
            </div>
          </div>

          <div className="panel" style={{ flex: 1 }}>
            <div className="panel-header">Execution Terminal</div>
            <div className="log-terminal">
              {log.length === 0 ? "IDLE. AWAITING TICKER DIFF..." : log.map((l, i) => <div key={i}>{l}</div>)}
            </div>
            <button
              className="button-trigger"
              onClick={handleRebalance}
              disabled={executing}
              style={{ marginTop: '1rem' }}
            >
              {executing ? "EXECUTING RECONCILIATION..." : "FORCE MANUAL RECONCILIATION"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
