import { useCallback, useEffect, useState } from 'react';
import './index.css';

const POLL_MS = 30_000;

function resolveApiBase() {
  const configured = import.meta.env.VITE_API_URL?.trim();
  if (configured) {
    return configured.replace(/\/+$/, '');
  }

  const { hostname, origin } = window.location;
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return 'http://localhost:8000';
  }

  return origin.replace(/\/+$/, '');
}

const API = resolveApiBase();

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
    } catch (exc) {
      setError(exc.message);
    } finally {
      setLoading(false);
    }
  }, [date]);

  useEffect(() => {
    const initialId = setTimeout(() => {
      void fetchData();
    }, 0);
    const intervalId = setInterval(() => {
      void fetchData();
    }, POLL_MS);

    return () => {
      clearTimeout(initialId);
      clearInterval(intervalId);
    };
  }, [fetchData]);

  return { data, error, loading, lastFetch, refresh: fetchData };
}

function useDates() {
  const [dates, setDates] = useState([]);

  useEffect(() => {
    fetch(`${API}/api/dates`)
      .then((res) => res.json())
      .then(setDates)
      .catch(() => {});
  }, []);

  return dates;
}

const asPct = (value) => (value != null ? `${(value * 100).toFixed(1)}%` : '--');
const asSigned = (value) => {
  if (value == null) return '--';
  return `${value > 0 ? '+' : ''}${value.toFixed(2)}`;
};
const asTimestamp = (value) => (value ? value.replace('T', ' ').slice(0, 19) : '--');

function mean(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function buildExecLog(data) {
  if (!data) return [];

  const { pipeline_run: run, portfolio, risk_actions } = data;
  const log = [];

  if (!run.phase1_complete) {
    log.push('[SYS]  No pipeline data yet. Run orchestrator/daily_pipeline.py.');
  } else {
    log.push(`[INFO] Phase 1 complete - ${run.stocks_complete}/${run.stocks_total} stocks analyzed`);
    if (run.error_tickers?.length) {
      log.push(`[WARN] ${run.error_tickers.join(', ')} fell back to defaults`);
    }
  }

  if (run.phase2_complete) {
    log.push(`[INFO] Phase 2 complete - style=${portfolio.trading_style?.toUpperCase() ?? '--'} cash=${asPct(portfolio.cash_reserve)}`);
    log.push(`[INFO] Target portfolio contains ${Object.keys(portfolio.weights).length} active position(s)`);
  } else if (run.phase1_complete) {
    log.push('[ERR]  Portfolio-Decision has not produced allocations yet');
  }

  if (run.phase3_complete) {
    log.push(`[INFO] Phase 3 complete - ${risk_actions.length} risk action(s) emitted`);
    risk_actions.forEach((action) => {
      log.push(`       ${action.ticker}: ${action.action} | pnl ${action.pnl != null ? action.pnl.toFixed(2) : '?'}%`);
    });
  } else if (run.phase2_complete) {
    log.push('[ERR]  Risk management did not finalize');
  }

  return log;
}

function deriveDashboard(data) {
  const { pipeline_run: run, portfolio, stock_analyses, compliance, risk_actions } = data;
  const analyses = Object.entries(stock_analyses);
  const complete = analyses.filter(([, entry]) => entry.sentiment && entry.technical && entry.fundamentals);
  const weights = Object.entries(portfolio.weights).sort((a, b) => b[1] - a[1]);
  const hasWeights = weights.length > 0;

  const averageSentiment = mean(complete.map(([, entry]) => entry.sentiment.sentiment_score ?? 0));
  const averageHealth = mean(complete.map(([, entry]) => entry.fundamentals.health_score ?? 50));
  const upCount = complete.filter(([, entry]) => entry.technical.direction === 'up').length;
  const downCount = complete.filter(([, entry]) => entry.technical.direction === 'down').length;
  const breadth = complete.length ? (upCount - downCount) / complete.length : 0;

  let regime = 'BALANCED';
  if (averageSentiment > 0.15 || breadth > 0.2) regime = 'RISK-ON';
  if (averageSentiment < -0.15 || breadth < -0.2) regime = 'DEFENSIVE';

  const strip = analyses.map(([ticker, entry]) => {
    const sentiment = entry.sentiment?.sentiment_score ?? 0;
    const direction = entry.technical?.direction === 'up' ? 0.35 : entry.technical?.direction === 'down' ? -0.35 : 0;
    const health = ((entry.fundamentals?.health_score ?? 50) - 50) / 100;
    const score = sentiment + direction + health;
    let tone = 'strip-neutral';
    if (score > 0.35) tone = 'strip-positive';
    if (score < -0.15) tone = 'strip-negative';
    return { ticker, score, tone };
  });

  const rows = (hasWeights ? weights : complete
    .map(([ticker, entry]) => {
      const sentiment = entry.sentiment.sentiment_score ?? 0;
      const direction = entry.technical.direction === 'up' ? 1 : entry.technical.direction === 'down' ? -1 : 0;
      const health = (entry.fundamentals.health_score ?? 50) / 100;
      return [ticker, Math.max(0.02, sentiment * 0.08 + direction * 0.04 + health * 0.05)];
    })
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
  ).map(([ticker, weight]) => {
    const entry = stock_analyses[ticker] ?? {};
    return {
      ticker,
      weight,
      sentiment: entry.sentiment?.sentiment_score ?? null,
      direction: entry.technical?.direction ?? 'sideways',
      health: entry.fundamentals?.health_score ?? null,
      gate: entry.technical?.gate ?? '--',
      theme: entry.sentiment?.key_theme ?? 'No narrative',
      sector: entry.fundamentals?.sector ?? '--',
    };
  });

  const sectorRows = Object.entries(compliance.sector_weights)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  const grossExposure = compliance.gross_exposure ?? 0;
  const topWeight = hasWeights ? weights[0][1] : 0;
  const qualityScore = run.stocks_total ? run.stocks_complete / run.stocks_total : 0;
  const errorCount = run.error_tickers?.length ?? 0;

  return {
    run,
    portfolio,
    compliance,
    riskActions: risk_actions,
    hasWeights,
    averageSentiment,
    averageHealth,
    breadth,
    regime,
    strip,
    rows,
    sectorRows,
    grossExposure,
    topWeight,
    qualityScore,
    errorCount,
  };
}

function Header({ model, dataDate, lastFetch, dates, selectedDate, setSelectedDate }) {
  return (
    <header className="topbar">
      <div>
        <h1 className="title">HAILMARY OEX / S&amp;P 100</h1>
        <div className="subtitle">
          <span>SYS_VER: PIPELINE MIRROR</span>
          <span>REGIME: {model.regime}</span>
          <span>STYLE: {model.portfolio.trading_style?.toUpperCase() ?? '--'}</span>
        </div>
      </div>

      <div className="topbar-right">
        <div className="live-tag">LIVE.ALPACA</div>
        <div className="timestamp-row">
          {dates.length > 0 && (
            <select
              className="terminal-select"
              value={selectedDate ?? ''}
              onChange={(event) => setSelectedDate(event.target.value || null)}
            >
              <option value="">Today</option>
              {dates.map((date) => (
                <option key={date} value={date}>{date}</option>
              ))}
            </select>
          )}
          <span>{dataDate}</span>
          <span>{asTimestamp(lastFetch)}</span>
        </div>
      </div>
    </header>
  );
}

function Metric({ label, value, tone = 'bright' }) {
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className={`metric-value ${tone}`}>{value}</div>
    </div>
  );
}

function SnapshotPanel({ model }) {
  const confidenceTone = model.averageSentiment > 0.1 ? 'positive' : model.averageSentiment < -0.1 ? 'negative' : 'bright';
  return (
    <section className="panel">
      <div className="panel-title">Pipeline &amp; Risk Snapshot</div>

      <div className="metric-grid two">
        <Metric label="Trading Style" value={model.portfolio.trading_style?.toUpperCase() ?? '--'} tone="bright" />
        <Metric label="Cash Reserve" value={asPct(model.portfolio.cash_reserve)} tone="info" />
        <Metric label="Gross Exposure" value={asPct(model.grossExposure)} tone="bright" />
        <Metric label="Top Position" value={asPct(model.topWeight)} tone="neutral" />
        <Metric label="Signal Breadth" value={asSigned(model.breadth)} tone={confidenceTone} />
        <Metric label="Risk Actions" value={String(model.riskActions.length)} tone={model.riskActions.length ? 'negative' : 'positive'} />
      </div>

      <div className="legend-block">
        <div className="legend-line">
          <span className="legend-key positive">PIPELINE CONFIDENCE</span>
          <span>{asSigned(model.averageSentiment)}</span>
        </div>
        <div className="legend-line">
          <span className="legend-key dim">UNIVERSE COMPLETION</span>
          <span>{model.run.stocks_complete}/{model.run.stocks_total}</span>
        </div>
        <div className="legend-line">
          <span className="legend-key negative">DEFAULT FALLBACKS</span>
          <span>{model.errorCount}</span>
        </div>
      </div>
    </section>
  );
}

function HealthPanel({ model }) {
  return (
    <section className="panel">
      <div className="panel-title">Model Health &amp; Validation</div>

      <div className="health-row">
        <span>Data Quality</span>
        <span className="bright">{asPct(model.qualityScore)}</span>
      </div>
      <div className="bar">
        <div className="bar-fill positive" style={{ width: `${Math.round(model.qualityScore * 100)}%` }} />
      </div>

      <div className="health-row">
        <span>Average Health Score</span>
        <span className="bright">{model.averageHealth.toFixed(1)}</span>
      </div>
      <div className="health-row">
        <span>Phase 2 Status</span>
        <span className={model.run.phase2_complete ? 'positive' : 'negative'}>
          {model.run.phase2_complete ? 'ALLOCATED' : 'PENDING'}
        </span>
      </div>
      <div className="health-row">
        <span>Phase 3 Status</span>
        <span className={model.run.phase3_complete ? 'positive' : 'negative'}>
          {model.run.phase3_complete ? 'RISK-CHECKED' : 'PENDING'}
        </span>
      </div>
      <div className="health-row">
        <span>Last Pipeline Entry</span>
        <span className="dim">{asTimestamp(model.run.last_updated)}</span>
      </div>
    </section>
  );
}

function RegimePanel({ model }) {
  const sentimentTone = model.averageSentiment > 0.1 ? 'positive' : model.averageSentiment < -0.1 ? 'negative' : 'bright';
  return (
    <section className="panel">
      <div className="panel-title split">
        <span>Composite Market Regime</span>
        <span className="dim">{model.regime}</span>
      </div>

      <div className="regime-grid">
        <Metric label="Average Sentiment" value={asSigned(model.averageSentiment)} tone={sentimentTone} />
        <Metric label="Average Health" value={model.averageHealth.toFixed(1)} tone="bright" />
        <Metric label="Up / Down Breadth" value={`${model.run.stocks_complete ? Math.round(((model.breadth + 1) / 2) * 100) : 0}%`} tone="neutral" />
        <Metric label="Active Positions" value={String(model.rows.length)} tone="info" />
      </div>

      <div className="section-label">Watchlist Signal Strip</div>
      <div className="signal-strip">
        {model.strip.map((entry) => (
          <div key={entry.ticker} className={`signal-cell ${entry.tone}`} title={`${entry.ticker}: ${entry.score.toFixed(2)}`} />
        ))}
      </div>
      <div className="strip-labels">
        <span className="positive">BULLISH</span>
        <span className="neutral">MIXED</span>
        <span className="negative">DEFENSIVE</span>
      </div>
    </section>
  );
}

function AllocationPanel({ model }) {
  return (
    <section className="panel tall">
      <div className="panel-title">Active Allocations &amp; Signal Stack</div>

      <table className="terminal-table">
        <thead>
          <tr>
            <th>Ticker</th>
            <th>Weight</th>
            <th>Sent.</th>
            <th>Dir.</th>
            <th>Health</th>
            <th>Gate</th>
          </tr>
        </thead>
        <tbody>
          {model.rows.map((row) => (
            <tr key={row.ticker}>
              <td className="bright">{row.ticker}</td>
              <td className="info">{asPct(row.weight)}</td>
              <td className={row.sentiment > 0.1 ? 'positive' : row.sentiment < -0.1 ? 'negative' : 'dim'}>
                {row.sentiment == null ? '--' : asSigned(row.sentiment)}
              </td>
              <td className={row.direction === 'up' ? 'positive' : row.direction === 'down' ? 'negative' : 'dim'}>
                {row.direction.toUpperCase()}
              </td>
              <td className="neutral">{row.health ?? '--'}</td>
              <td className="dim gate-cell">{row.gate}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="section-divider" />

      <div className="section-label">Narrative Drivers</div>
      <div className="narrative-list">
        {model.rows.slice(0, 4).map((row) => (
          <div key={row.ticker} className="narrative-row">
            <span className="bright">{row.ticker}</span>
            <span className="dim">{row.theme}</span>
          </div>
        ))}
      </div>

      <div className="section-divider" />

      <div className="section-label">Sector Exposure</div>
      <div className="sector-list">
        {model.sectorRows.length === 0 ? (
          <div className="dim">No portfolio exposure yet.</div>
        ) : (
          model.sectorRows.map(([sector, weight]) => (
            <div key={sector} className="health-row">
              <span>{sector}</span>
              <span className="bright">{asPct(weight)}</span>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

function statusTone(ok, warn = false) {
  if (!ok) return 'negative';
  if (warn) return 'neutral';
  return 'positive';
}

function CompliancePanel({ model }) {
  const singleNameOk = model.topWeight <= model.compliance.single_name_max;
  const sectorOk = !model.compliance.violations.some((entry) => entry.includes('sector cap'));
  const grossOk = model.grossExposure <= 1;

  return (
    <section className="panel">
      <div className="panel-title">Pre-Trade Compliance Checks</div>

      <div className="health-row">
        <span>SINGLE-NAME CAP ({asPct(model.compliance.single_name_max)})</span>
        <span className={statusTone(singleNameOk)}>{singleNameOk ? 'PASS' : 'WARN'} [{asPct(model.topWeight)}]</span>
      </div>
      <div className="health-row">
        <span>SECTOR CAP ({asPct(model.compliance.sector_max)})</span>
        <span className={statusTone(sectorOk, !sectorOk)}>{sectorOk ? 'PASS' : 'WARN'} [{model.sectorRows[0] ? asPct(model.sectorRows[0][1]) : '--'}]</span>
      </div>
      <div className="health-row">
        <span>GROSS EXP. (100%)</span>
        <span className={statusTone(grossOk)}>{grossOk ? 'PASS' : 'WARN'} [{asPct(model.grossExposure)}]</span>
      </div>

      {model.compliance.violations.length > 0 && (
        <div className="violations">
          {model.compliance.violations.map((item) => (
            <div key={item} className="negative">{item}</div>
          ))}
        </div>
      )}
    </section>
  );
}

function ExecutionPanel({ model, lastFetch }) {
  return (
    <section className="panel">
      <div className="panel-title">Execution Subsystem</div>

      <div className="health-row">
        <span>ROUTING</span>
        <span className="bright">ALPACA (PAPER)</span>
      </div>
      <div className="health-row">
        <span>RISK STYLE</span>
        <span className="bright">{model.portfolio.trading_style?.toUpperCase() ?? '--'}</span>
      </div>
      <div className="health-row">
        <span>CASH RESERVE</span>
        <span className="info">{asPct(model.portfolio.cash_reserve)}</span>
      </div>
      <div className="health-row">
        <span>OPEN POSITIONS</span>
        <span className="bright">{model.rows.length}</span>
      </div>
      <div className="health-row">
        <span>LAST REFRESH</span>
        <span className="dim">{asTimestamp(lastFetch)}</span>
      </div>
    </section>
  );
}

function LogPanel({ lines, executing, onTrigger }) {
  return (
    <section className="panel fill">
      <div className="panel-title split">
        <span>Execution Log</span>
        <button className="action-link" onClick={onTrigger} disabled={executing}>
          {executing ? '[EXECUTING]' : '[TRIGGER REBALANCE]'}
        </button>
      </div>

      <div className="log-box">
        {lines.length === 0 ? (
          <div className="dim">IDLE. STANDBY FOR SERVER DATA...</div>
        ) : (
          lines.map((line, index) => (
            <div
              key={`${line}-${index}`}
              className={
                line.startsWith('[ERR') ? 'negative'
                  : line.startsWith('[WARN') ? 'neutral'
                    : line.startsWith('[SYS') ? 'info'
                      : line.startsWith('[INFO]') ? 'bright'
                        : 'dim'
              }
            >
              {line}
            </div>
          ))
        )}
      </div>
    </section>
  );
}

export default function App() {
  const [selectedDate, setSelectedDate] = useState(null);
  const [execMessages, setExecMessages] = useState([]);
  const [executing, setExecuting] = useState(false);
  const dates = useDates();
  const { data, error, loading, lastFetch, refresh } = useStatus(selectedDate);

  const handleRebalance = () => {
    setExecuting(true);
    setExecMessages((current) => [...current, '[SYS]  Dashboard does not send live orders directly.']);
    setTimeout(() => {
      setExecMessages((current) => [...current, '[INFO] Run orchestrator/daily_pipeline.py to execute target weights.']);
      setExecuting(false);
    }, 1200);
  };

  if (loading) {
    return (
      <div className="dashboard-shell centered">
        <div className="panel offline-panel">
          <div className="panel-title">Connecting</div>
          <div className="bright">WAITING FOR BACKEND...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard-shell centered">
        <div className="panel offline-panel">
          <div className="panel-title">Backend Offline</div>
          <div className="negative">{error}</div>
          <div className="dim">Start the API from the project root:</div>
          <div className="bright">uvicorn backend.server:app --reload --port 8000</div>
          <button className="retry-button" onClick={refresh}>Retry</button>
        </div>
      </div>
    );
  }

  const model = deriveDashboard(data);
  const execLog = [...buildExecLog(data), ...execMessages];

  return (
    <div className="dashboard-shell">
      <Header
        model={model}
        dataDate={data.date}
        lastFetch={lastFetch}
        dates={dates}
        selectedDate={selectedDate}
        setSelectedDate={setSelectedDate}
      />

      <div className="dashboard-grid">
        <div className="column column-left">
          <SnapshotPanel model={model} />
          <HealthPanel model={model} />
        </div>

        <div className="column column-center">
          <RegimePanel model={model} />
          <AllocationPanel model={model} />
        </div>

        <div className="column column-right">
          <CompliancePanel model={model} />
          <ExecutionPanel model={model} lastFetch={lastFetch} />
          <LogPanel lines={execLog} executing={executing} onTrigger={handleRebalance} />
        </div>
      </div>
    </div>
  );
}
