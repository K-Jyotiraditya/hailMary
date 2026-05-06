import React, { useState } from 'react';
import './index.css';

// Synthetic structural data mirroring live algorithm state
const SYS_STATE = {
  modelHealth: {
    backtestIC: 0.0434,
    liveIC: 0.0392,
    featureDrift: 0.012,
    nextRetrain: "14 Days",
    lastRetrain: "2026-04-20 23:00Z",
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
    scaler: 0.84,
    stripHistory: Array.from({ length: 60 }, (_, i) => Math.random() > 0.8 ? "danger" : Math.random() > 0.4 ? "warning" : "safe")
  },
  performance: {
    maxDD: "-3.41%",
    currentDD: "-0.84%",
    rollingRankIC: "0.045", // Replaced Hit Rate
    sharpe60d: 1.12,
    mtd: "+1.4%",
    ytd: "+8.2%"
  },
  positions: [
    { ticker: "AAPL", exp: "15%", riskAttr: "4.2%", attr: "vol_60(-1.2) mom_20(+2.1)", drift: 0.01 },
    { ticker: "MSFT", exp: "10%", riskAttr: "2.1%", attr: "mean_abs_ic(+1.5) MACD(+0.8)", drift: 0.02 },
    { ticker: "NVDA", exp: "12%", riskAttr: "8.5%", attr: "dollar_vol(+2.4) vs_vwap(+1.1)", drift: -0.01 }
  ],
  sectors: [
    { name: "Information Technology", val: "37%" },
    { name: "Financials", val: "15%" },
    { name: "Healthcare", val: "12%" }
  ],
  execution: {
    routing: "ALPACA (US EQ)",
    cashPct: "63%", // Replaced Margin Avail
    slippageModel: "ADV Square-root (Est. 2bps)",
    lastFetch: "2026-05-06T15:55:01Z"
  },
  compliance: {
    singleNameMax: { limit: "15%", current: "15%", pass: true },
    sectorCap: { limit: "35%", current: "37%", pass: false }, // Intentional warning
    grossExp: { limit: "100%", current: "37%", pass: true }
  }
};

const EquityCurve = () => (
  <div style={{ position: 'relative', width: '100%', height: '100px', marginTop: '0.8rem' }}>
    <div style={{ position: 'absolute', top: 0, right: 0, fontSize: '0.65rem', color: 'var(--positive)' }}>SOTA ENGINE</div>
    <div style={{ position: 'absolute', top: '15px', right: 0, fontSize: '0.65rem', color: '#666' }}>S&P 100 BMRK</div>
    <div style={{ position: 'absolute', top: '30px', right: 0, fontSize: '0.65rem', color: 'var(--negative)' }}>DRAWDOWN</div>
    <svg viewBox="0 0 300 100" style={{ width: '100%', height: '100%', borderBottom: '1px solid var(--border-subtle)' }}>
      {/* Drawdown shading */}
      <path d="M0,100 L0,80 L20,82 L40,85 L80,50 L100,55 L150,20 L200,30 L250,10 L300,5 L300,100 Z" fill="rgba(255, 51, 102, 0.15)" />

      {/* Strategy Curve */}
      <polyline points="0,80 20,82 40,85 80,50 100,55 150,20 200,30 250,10 300,5" fill="none" stroke="var(--positive)" strokeWidth="1.5" />

      {/* Benchmark Curve */}
      <polyline points="0,95 20,90 40,92 80,75 100,85 150,60 200,70 250,65 300,55" fill="none" stroke="#666" strokeWidth="1" strokeDasharray="4 4" />
    </svg>
  </div>
);

const CpcvHistogram = () => (
  <div style={{ position: 'relative', display: 'flex', alignItems: 'flex-end', height: '40px', gap: '2px', marginTop: '0.5rem' }}>
    <div style={{ position: 'absolute', top: '-15px', left: '60%', fontSize: '0.65rem', color: 'var(--info)' }}>DSR: 1.84</div>
    {[1.1, 1.3, 1.8, 3.4, 5.1, 6.5, 5.2, 3.8, 2.1, 1.4, 0.9].map((h, i) => (
      <div key={i} style={{ flex: 1, background: 'var(--border-strong)', height: `${h * 5}px` }}></div>
    ))}
    <div style={{ position: 'absolute', left: '62%', width: '1px', height: '45px', background: 'var(--info)' }} />
  </div>
);

function App() {
  const [executing, setExecuting] = useState(false);
  const [log, setLog] = useState([]);

  const handleRebalance = () => {
    setExecuting(true);
    setLog(["[SYS] Dispatching API Call to Execution Node..."]);
    setTimeout(() => setLog(p => [...p, "[API] NAV: 100000.0 | Cash: 63000.0"]), 500);
    setTimeout(() => setLog(p => [...p, "[EXEC] AAPL Diff: $15,000 -> Constructing Bracket..."]), 1200);
    setTimeout(() => setLog(p => [...p, "[FILL] NVDA 124 shares @ MKT | Realized Slippage: 1.4bps"]), 2200);
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
                <div className="metric-label">CURRENT vs MAX DD</div>
                <div className="metric-val text-white">{SYS_STATE.performance.currentDD} / <span className="text-dim">{SYS_STATE.performance.maxDD}</span></div>
              </div>
              <div>
                <div className="metric-label">ROLLING 60D SHARPE</div>
                <div className="metric-val text-info">{SYS_STATE.performance.sharpe60d}</div>
              </div>
              <div>
                <div className="metric-label">MTD / YTD RET</div>
                <div className="metric-val text-pos">{SYS_STATE.performance.mtd} / {SYS_STATE.performance.ytd}</div>
              </div>
              <div>
                <div className="metric-label">ROLLING RANK IC</div>
                <div className="metric-val text-white">{SYS_STATE.performance.rollingRankIC}</div>
              </div>
            </div>
            <EquityCurve />
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
              <span className="text-dim">PROB. OVERFIT (PBO)</span>
              <span className="text-white">{SYS_STATE.modelHealth.pbo}</span>
            </div>

            <div style={{ marginTop: '0.8rem' }}>
              <span className="text-dim" style={{ fontSize: '0.65rem' }}>CPCV PATH DISTRIBUTION (SHARPE)</span>
              <CpcvHistogram />
            </div>

            <div className="flex-between" style={{ marginTop: '0.8rem' }}>
              <span className="text-dim">LAST RETRAIN</span>
              <span className="text-dim">{SYS_STATE.modelHealth.lastRetrain}</span>
            </div>
            <div className="flex-between">
              <span className="text-dim">NEXT SCHEDULED</span>
              <span className="text-white">{SYS_STATE.modelHealth.nextRetrain}</span>
            </div>
          </div>
        </div>

        {/* MIDDLE COLUMN: REGIME & POSITIONS */}
        <div className="flex-col">
          <div className="panel">
            <div className="panel-header flex-between">
              <span>Composite Market Regime</span>
              <span>TARGET VOL: {SYS_STATE.regime.targetVol * 100}% | REALIZED: {(SYS_STATE.regime.realizedVol * 100).toFixed(1)}%</span>
            </div>
            <div className="grid-3" style={{ marginBottom: '0.5rem' }}>
              <div>
                <div className="metric-label">SMA-200 DISTANCE</div>
                <div className="metric-val text-neg">{SYS_STATE.regime.sma200Dist}</div>
              </div>
              <div>
                <div className="metric-label">ADX INDICATION ({SYS_STATE.regime.adxScore})</div>
                <div style={{ fontSize: '0.7rem', color: '#666', marginTop: '2px' }}>{SYS_STATE.regime.adxLabel}</div>
              </div>
              <div>
                <div className="metric-label">COMPOSITE MULTIPLIER</div>
                <div className="metric-val text-info">{(SYS_STATE.regime.scaler * SYS_STATE.regime.combinedScore).toFixed(2)}x</div>
              </div>
            </div>

            <div className="metric-label" style={{ marginTop: '0.2rem', marginBottom: '0.1rem' }}>TRAILING 60-DAY REGIME STATE TRANSITIONS</div>
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
                  <th>Ticker</th>
                  <th>Weight</th>
                  <th>Vol Contrib</th>
                  <th>Feature Drivers (Attribution)</th>
                  <th>Drift</th>
                </tr>
              </thead>
              <tbody>
                {SYS_STATE.positions.map((p, i) => (
                  <tr key={i}>
                    <td className="text-white font-weight-700">{p.ticker}</td>
                    <td className="text-info">{p.exp}</td>
                    <td className="text-neutral">{p.riskAttr}</td>
                    <td className="text-dim">{p.attr}</td>
                    <td className={p.drift > 0 ? "text-pos" : "text-neg"}>{p.drift}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div style={{ marginTop: '1rem', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.5rem' }}>
              <div className="metric-label">SECTOR EXPOSURE (% Portfolio)</div>
              {SYS_STATE.sectors.map((s, i) => (
                <div className="flex-between" key={i} style={{ marginTop: '2px' }}>
                  <span className="text-dim">{s.name}</span>
                  <span className="text-white">{s.val}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: OP EXECUTION */}
        <div className="flex-col">
          <div className="panel">
            <div className="panel-header">Pre-Trade Compliance Checks</div>
            <div className="flex-between">
              <span className="text-dim">SINGLE-NAME CAP ({SYS_STATE.compliance.singleNameMax.limit})</span>
              <span className={SYS_STATE.compliance.singleNameMax.pass ? "text-pos" : "text-neg"}>{SYS_STATE.compliance.singleNameMax.pass ? "PASS" : "WARN"} [{SYS_STATE.compliance.singleNameMax.current}]</span>
            </div>
            <div className="flex-between">
              <span className="text-dim">SECTOR CAP ({SYS_STATE.compliance.sectorCap.limit})</span>
              <span className={SYS_STATE.compliance.sectorCap.pass ? "text-pos" : "text-neg"}>{SYS_STATE.compliance.sectorCap.pass ? "PASS" : "WARN"} [{SYS_STATE.compliance.sectorCap.current}]</span>
            </div>
            <div className="flex-between">
              <span className="text-dim">GROSS EXP. ({SYS_STATE.compliance.grossExp.limit})</span>
              <span className={SYS_STATE.compliance.grossExp.pass ? "text-pos" : "text-neg"}>{SYS_STATE.compliance.grossExp.pass ? "PASS" : "WARN"} [{SYS_STATE.compliance.grossExp.current}]</span>
            </div>
          </div>

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
              <span className="text-dim">CASH RESERVE</span>
              <span className="text-white">{SYS_STATE.execution.cashPct}</span>
            </div>
          </div>

          <div className="panel" style={{ flex: 1 }}>
            <div className="panel-header flex-between">
              <span>Execution Log</span>
              <a href="#" onClick={(e) => { e.preventDefault(); handleRebalance() }} className="text-dim" style={{ fontFamily: 'inherit', fontSize: '0.65rem', textDecoration: 'none' }}>[ADMIN: EXECUTE PREDICTIONS]</a>
            </div>
            <div className="log-terminal">
              {log.length === 0 ? "IDLE. STANDBY FOR SERVER WEBSOCKET..." : log.map((l, i) => <div key={i}>{l}</div>)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
