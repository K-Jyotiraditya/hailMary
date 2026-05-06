import { useState, useEffect, useCallback } from 'react';
import './index.css';

const API = 'http://localhost:8000';
const POLL_MS = 30_000;

// ── Data hooks ────────────────────────────────────────────────────────────

function useStatus(date) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastFetch, setLastFetch] = useState(null);

  const fetchData = useCallback(async () => {
    const url = date ? `${API}/api/logs/${date}` : `${API}/api/status`;
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
      setError(null);
      setLastFetch(new Date().toISOString());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [date]);

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, POLL_MS);
    return () => clearInterval(id);
  }, [fetchData]);

  return { data, error, loading, lastFetch, refresh: fetchData };
}

function useDates() {
  const [dates, setDates] = useState([]);
  useEffect(() => {
    fetch(`${API}/api/dates`).then(r => r.json()).then(setDates).catch(() => {});
  }, []);
  return dates;
}

// ── Formatters ────────────────────────────────────────────────────────────

const fmtPct = v => v != null ? `${(v * 100).toFixed(1)}%` : '—';
const fmtTs = ts => ts ? ts.replace('T', ' ').slice(0, 19) : '—';

function fmtSentiment(v) {
  if (v == null) return { text: '—', cls: 'text-dim' };
  if (v > 0.1)  return { text: `+${v.toFixed(2)}`, cls: 'text-pos' };
  if (v < -0.1) return { text: v.toFixed(2),        cls: 'text-neg' };
  return { text: v.toFixed(2), cls: 'text-dim' };
}

function fmtDirection(d) {
  if (!d || d === 'sideways') return { text: 'FLAT', cls: 'text-dim' };
  if (d === 'up')   return { text: 'UP',   cls: 'text-pos' };
  if (d === 'down') return { text: 'DOWN', cls: 'text-neg' };
  return { text: d.toUpperCase(), cls: 'text-dim' };
}

function fmtHealth(h) {
  if (h == null) return { text: '—', cls: 'text-dim' };
  if (h >= 65)   return { text: String(h), cls: 'text-pos' };
  if (h <= 35)   return { text: String(h), cls: 'text-neg' };
  return { text: String(h), cls: 'text-neutral' };
}

function phaseTag(ok) {
  return ok
    ? <span className="text-pos">COMPLETE</span>
    : <span className="text-neg">NOT RUN</span>;
}

// Derive a composite signal score for ranking Phase-1-only results
function compositeScore(d) {
  const sent = d.sentiment?.sentiment_score ?? 0;
  const dir  = d.technical?.direction === 'up' ? 1 : d.technical?.direction === 'down' ? -1 : 0;
  const hlth = ((d.fundamentals?.health_score ?? 50) - 50) / 100;
  return sent + dir + hlth;
}

// ── Sub-components ────────────────────────────────────────────────────────

function PipelineStatus({ run }) {
  return (
    <div className="panel">
      <div className="panel-header">Pipeline Run Status — {run.date ?? '—'}</div>

      <div className="flex-between">
        <span className="text-dim">PHASE 1 — Per-Stock Agents</span>
        {phaseTag(run.phase1_complete)}
      </div>
      <div className="flex-between" style={{ paddingLeft: '1rem', fontSize: '0.72rem' }}>
        <span className="text-dim">Stocks complete</span>
        <span className={run.stocks_complete === run.stocks_total ? 'text-pos' : 'text-neutral'}>
          {run.stocks_complete} / {run.stocks_total}
        </span>
      </div>

      <div className="flex-between">
        <span className="text-dim">PHASE 2 — Portfolio Decision</span>
        {phaseTag(run.phase2_complete)}
      </div>

      <div className="flex-between">
        <span className="text-dim">PHASE 3 — Risk Management</span>
        {phaseTag(run.phase3_complete)}
      </div>

      <div className="flex-between" style={{ marginTop: '0.25rem', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.5rem' }}>
        <span className="text-dim">LOG ENTRIES</span>
        <span className="text-white">{run.total_entries}</span>
      </div>
      <div className="flex-between">
        <span className="text-dim">LAST ENTRY</span>
        <span className="text-dim" style={{ fontSize: '0.7rem' }}>{fmtTs(run.last_updated)}</span>
      </div>
    </div>
  );
}

function StockTable({ analyses }) {
  const tickers = Object.keys(analyses);
  if (!tickers.length)
    return <div className="text-dim" style={{ fontSize: '0.75rem' }}>No per-stock data yet.</div>;

  return (
    <div style={{ overflowY: 'auto', maxHeight: '300px' }}>
      <table>
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Sent.</th>
            <th>Dir.</th>
            <th>Gate</th>
            <th>Health</th>
          </tr>
        </thead>
        <tbody>
          {tickers.map(ticker => {
            const d    = analyses[ticker];
            const sent = fmtSentiment(d.sentiment?.sentiment_score);
            const dir  = fmtDirection(d.technical?.direction);
            const hlth = fmtHealth(d.fundamentals?.health_score);
            const ok   = d.sentiment && d.technical && d.fundamentals;
            return (
              <tr key={ticker} style={!ok ? { opacity: 0.45 } : {}}>
                <td style={{ fontWeight: 700, color: 'var(--text-highlight)' }}>{ticker}</td>
                <td className={sent.cls}>{sent.text}</td>
                <td className={dir.cls}>{dir.text}</td>
                <td className="text-dim" style={{ fontSize: '0.68rem' }}>{d.technical?.gate ?? '—'}</td>
                <td className={hlth.cls}>{hlth.text}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function SignalRanking({ analyses }) {
  const ranked = Object.entries(analyses)
    .map(([ticker, d]) => ({ ticker, d, score: compositeScore(d) }))
    .filter(x => x.d.sentiment && x.d.technical && x.d.fundamentals)
    .sort((a, b) => b.score - a.score);

  if (!ranked.length)
    return <div className="text-dim" style={{ fontSize: '0.75rem' }}>No complete stock data to rank.</div>;

  return (
    <div style={{ overflowY: 'auto', maxHeight: '260px' }}>
      <div className="text-dim" style={{ fontSize: '0.68rem', marginBottom: '0.4rem' }}>
        Ranked by composite signal (sentiment + direction + fundamentals).
        Portfolio-Decision agent has not run — these are raw Phase 1 signals only.
      </div>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Ticker</th>
            <th>Score</th>
            <th>Sent.</th>
            <th>Dir.</th>
            <th>Health</th>
          </tr>
        </thead>
        <tbody>
          {ranked.map(({ ticker, d, score }, i) => {
            const sent = fmtSentiment(d.sentiment?.sentiment_score);
            const dir  = fmtDirection(d.technical?.direction);
            const hlth = fmtHealth(d.fundamentals?.health_score);
            return (
              <tr key={ticker}>
                <td className="text-dim">{i + 1}</td>
                <td style={{ fontWeight: 700, color: 'var(--text-highlight)' }}>{ticker}</td>
                <td className={score > 0.5 ? 'text-pos' : score < -0.5 ? 'text-neg' : 'text-dim'}>
                  {score > 0 ? '+' : ''}{score.toFixed(2)}
                </td>
                <td className={sent.cls}>{sent.text}</td>
                <td className={dir.cls}>{dir.text}</td>
                <td className={hlth.cls}>{hlth.text}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function WeightsTable({ weights, analyses }) {
  const entries = Object.entries(weights).sort((a, b) => b[1] - a[1]);
  if (!entries.length) return null;

  return (
    <table>
      <thead>
        <tr>
          <th>Ticker</th>
          <th>Weight</th>
          <th>Dir.</th>
          <th>Sent.</th>
          <th>Health</th>
        </tr>
      </thead>
      <tbody>
        {entries.map(([ticker, w]) => {
          const d    = analyses[ticker] || {};
          const dir  = fmtDirection(d.technical?.direction);
          const sent = fmtSentiment(d.sentiment?.sentiment_score);
          const hlth = fmtHealth(d.fundamentals?.health_score);
          return (
            <tr key={ticker}>
              <td style={{ fontWeight: 700, color: 'var(--text-highlight)' }}>{ticker}</td>
              <td className="text-info">{fmtPct(w)}</td>
              <td className={dir.cls}>{dir.text}</td>
              <td className={sent.cls}>{sent.text}</td>
              <td className={hlth.cls}>{hlth.text}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function CompliancePanel({ compliance, hasWeights }) {
  const { sector_weights, single_name_max, sector_max, gross_exposure, violations } = compliance;

  if (!hasWeights) {
    return (
      <>
        <div className="flex-between">
          <span className="text-dim">SINGLE-NAME CAP</span>
          <span className="text-white">{fmtPct(single_name_max)}</span>
        </div>
        <div className="flex-between">
          <span className="text-dim">SECTOR CAP</span>
          <span className="text-white">{fmtPct(sector_max)}</span>
        </div>
        <div className="text-dim" style={{ fontSize: '0.7rem', marginTop: '0.3rem' }}>
          Checks active once Portfolio-Decision runs.
        </div>
      </>
    );
  }

  return (
    <>
      <div className="flex-between">
        <span className="text-dim">GROSS EXPOSURE</span>
        <span className="text-white">{fmtPct(gross_exposure)}</span>
      </div>
      {Object.entries(sector_weights)
        .sort((a, b) => b[1] - a[1])
        .map(([sector, w]) => {
          const over = w > sector_max;
          return (
            <div className="flex-between" key={sector}>
              <span className="text-dim" style={{ fontSize: '0.72rem' }}>{sector.toUpperCase()}</span>
              <span className={over ? 'text-neg' : 'text-white'}>
                {fmtPct(w)} {over ? 'WARN' : 'OK'}
              </span>
            </div>
          );
        })}
      {violations.length > 0 && (
        <div style={{ marginTop: '0.5rem', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.5rem' }}>
          {violations.map((v, i) => (
            <div key={i} className="text-neg" style={{ fontSize: '0.7rem' }}>{v}</div>
          ))}
        </div>
      )}
      {violations.length === 0 && (
        <div className="text-pos" style={{ fontSize: '0.7rem', marginTop: '0.3rem' }}>
          All constraints satisfied.
        </div>
      )}
    </>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────

export default function App() {
  const [selectedDate, setSelectedDate] = useState(null);
  const dates = useDates();
  const { data, error, loading, lastFetch, refresh } = useStatus(selectedDate);
  const [execLog, setExecLog] = useState([]);
  const [executing, setExecuting] = useState(false);

  // Build pipeline event log from API data whenever data updates
  useEffect(() => {
    if (!data) return;
    const { pipeline_run: run, portfolio, risk_actions } = data;
    const log = [];

    if (!run.phase1_complete) {
      log.push('[SYS]  No pipeline data. Run orchestrator/daily_pipeline.py to start.');
    } else {
      log.push(`[INFO] Phase 1 complete — ${run.stocks_complete}/${run.stocks_total} stocks analyzed`);
      if (run.stocks_complete < run.stocks_total) {
        const missing = run.stocks_total - run.stocks_complete;
        log.push(`[WARN] ${missing} stock(s) had a failed agent call (Fundamentals likely)`);
      }
    }

    if (run.phase2_complete) {
      const style = portfolio.trading_style?.toUpperCase() ?? '?';
      log.push(`[INFO] Phase 2 complete — style=${style}, cash=${fmtPct(portfolio.cash_reserve)}`);
      const nPositions = Object.keys(portfolio.weights).length;
      log.push(`[INFO] ${nPositions} position(s) in target portfolio`);
    } else if (run.phase1_complete) {
      log.push('[ERR]  Phase 2 did not run — Portfolio-Decision agent missing or crashed');
    }

    if (run.phase3_complete) {
      log.push(`[INFO] Phase 3 complete — ${risk_actions.length} risk action(s)`);
      risk_actions.forEach(a =>
        log.push(`       ${a.ticker}: ${a.action} | PnL ${a.pnl != null ? a.pnl.toFixed(2) : '?'}%`)
      );
    } else if (run.phase2_complete) {
      log.push('[ERR]  Phase 3 did not run — Risk-Management agent missing or crashed');
    }

    setExecLog(log);
  }, [data]);

  const handleRebalance = () => {
    setExecuting(true);
    setExecLog(p => [...p, '[SYS]  Execution via dashboard not supported.']);
    setTimeout(() => {
      setExecLog(p => [...p, '[INFO] Run orchestrator/daily_pipeline.py for live execution.']);
      setExecuting(false);
    }, 1200);
  };

  // ── Loading / Error screens ──────────────────────────────────────────────

  if (loading) {
    return (
      <div className="dashboard-container" style={{ paddingTop: '3rem', textAlign: 'center' }}>
        <div className="text-dim">CONNECTING TO BACKEND...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard-container" style={{ paddingTop: '3rem' }}>
        <div className="panel" style={{ maxWidth: '560px' }}>
          <div className="panel-header">Backend Offline</div>
          <div className="text-neg" style={{ marginBottom: '0.5rem' }}>{error}</div>
          <div className="text-dim" style={{ fontSize: '0.75rem' }}>
            Start the server from the project root:
          </div>
          <div className="text-white" style={{ fontSize: '0.75rem', margin: '0.4rem 0 1rem' }}>
            uvicorn backend.server:app --reload --port 8000
          </div>
          <button className="button-trigger" onClick={refresh}>Retry</button>
        </div>
      </div>
    );
  }

  const { pipeline_run: run, portfolio, stock_analyses, compliance, risk_actions } = data;
  const hasWeights = Object.keys(portfolio.weights).length > 0;
  const styleColor = portfolio.trading_style === 'aggressive' ? 'text-neg'
                   : portfolio.trading_style === 'conservative' ? 'text-info'
                   : 'text-neutral';

  // ── Main layout ──────────────────────────────────────────────────────────

  return (
    <div className="dashboard-container">

      {/* Header */}
      <header className="terminal-header">
        <div>
          <h1 className="terminal-title">HAILMARY OEX / S&amp;P 100</h1>
          <div style={{ display: 'flex', gap: '1.2rem', alignItems: 'center', marginTop: '0.2rem', fontSize: '0.7rem' }}>
            <span>P1: {run.phase1_complete ? <span className="text-pos">OK</span> : <span className="text-neg">--</span>}</span>
            <span>P2: {run.phase2_complete ? <span className="text-pos">OK</span> : <span className="text-neg">--</span>}</span>
            <span>P3: {run.phase3_complete ? <span className="text-pos">OK</span> : <span className="text-neg">--</span>}</span>
            <span className="text-dim">{run.stocks_complete}/{run.stocks_total} stocks</span>
            <span className="text-dim">{run.total_entries} entries</span>
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <span className="text-info" style={{ fontSize: '0.8rem' }}>LIVE.ALPACA.PAPER</span>
          <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end', marginTop: '0.2rem', alignItems: 'center' }}>
            {dates.length > 0 && (
              <select
                value={selectedDate ?? ''}
                onChange={e => setSelectedDate(e.target.value || null)}
                style={{ background: '#111', color: '#666', border: '1px solid #333', fontFamily: 'inherit', fontSize: '0.7rem', padding: '2px 4px' }}
              >
                <option value="">Today</option>
                {dates.map(d => <option key={d} value={d}>{d}</option>)}
              </select>
            )}
            <span className="text-dim" style={{ fontSize: '0.65rem' }}>Updated {fmtTs(lastFetch)}</span>
          </div>
        </div>
      </header>

      <div className="grid-main">

        {/* ── LEFT COLUMN ── */}
        <div className="flex-col">
          <PipelineStatus run={{ ...run, date: data.date }} />

          <div className="panel" style={{ flex: 1 }}>
            <div className="panel-header">Per-Stock Analysis — Phase 1</div>
            <StockTable analyses={stock_analyses} />
          </div>
        </div>

        {/* ── MIDDLE COLUMN ── */}
        <div className="flex-col">

          {/* Portfolio decision */}
          <div className="panel">
            <div className="panel-header flex-between">
              <span>Portfolio Decision</span>
              {portfolio.trading_style && (
                <span className={styleColor}>{portfolio.trading_style.toUpperCase()}</span>
              )}
            </div>

            {run.phase2_complete ? (
              <>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                  <div>
                    <div className="metric-label">Trading Style</div>
                    <div className="metric-val">{portfolio.trading_style ?? '—'}</div>
                  </div>
                  <div>
                    <div className="metric-label">Cash Reserve</div>
                    <div className="metric-val text-info">{fmtPct(portfolio.cash_reserve)}</div>
                  </div>
                </div>
                {portfolio.rationale && (
                  <div style={{ marginTop: '0.5rem', fontSize: '0.72rem', color: '#888', lineHeight: 1.45, borderTop: '1px solid var(--border-subtle)', paddingTop: '0.5rem' }}>
                    {portfolio.rationale}
                  </div>
                )}
              </>
            ) : (
              <div style={{ padding: '0.5rem 0' }}>
                <div className="text-dim" style={{ fontSize: '0.75rem' }}>
                  Phase 2 has not run — no portfolio decision available.
                </div>
                <div className="text-dim" style={{ fontSize: '0.65rem', marginTop: '0.3rem' }}>
                  Cash: 100% | Style: pending
                </div>
              </div>
            )}
          </div>

          {/* Weights or signal ranking */}
          <div className="panel" style={{ flex: 1 }}>
            <div className="panel-header">
              {hasWeights ? 'Target Portfolio Weights' : 'Phase 1 Signal Ranking (No Portfolio Decision Yet)'}
            </div>
            {hasWeights
              ? <WeightsTable weights={portfolio.weights} analyses={stock_analyses} />
              : <SignalRanking analyses={stock_analyses} />
            }
          </div>
        </div>

        {/* ── RIGHT COLUMN ── */}
        <div className="flex-col">

          {/* Compliance */}
          <div className="panel">
            <div className="panel-header">Pre-Trade Compliance</div>
            <CompliancePanel compliance={compliance} hasWeights={hasWeights} />
          </div>

          {/* Execution subsystem */}
          <div className="panel">
            <div className="panel-header">Execution Subsystem</div>
            <div className="flex-between">
              <span className="text-dim">ROUTING</span>
              <span className="text-white">ALPACA (PAPER)</span>
            </div>
            <div className="flex-between">
              <span className="text-dim">CASH RESERVE</span>
              <span className="text-white">{fmtPct(portfolio.cash_reserve)}</span>
            </div>
            <div className="flex-between">
              <span className="text-dim">OPEN POSITIONS</span>
              <span className="text-white">{Object.keys(portfolio.weights).length}</span>
            </div>
            <div className="flex-between">
              <span className="text-dim">RISK ACTIONS</span>
              <span className={risk_actions.length > 0 ? 'text-neutral' : 'text-dim'}>
                {risk_actions.length}
              </span>
            </div>
          </div>

          {/* Pipeline log */}
          <div className="panel" style={{ flex: 1 }}>
            <div className="panel-header flex-between">
              <span>Pipeline Log</span>
              <a
                href="#"
                onClick={e => { e.preventDefault(); if (!executing) handleRebalance(); }}
                className="text-dim"
                style={{ fontFamily: 'inherit', fontSize: '0.65rem', textDecoration: 'none' }}
              >
                [TRIGGER REBALANCE]
              </a>
            </div>
            <div className="log-terminal">
              {execLog.length === 0
                ? 'IDLE. NO DATA.'
                : execLog.map((line, i) => (
                  <div
                    key={i}
                    style={{
                      color: line.startsWith('[ERR')  ? 'var(--negative)'
                           : line.startsWith('[WARN') ? 'var(--neutral)'
                           : line.startsWith('[SYS')  ? 'var(--info)'
                           : '#888',
                    }}
                  >
                    {line}
                  </div>
                ))
              }
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
