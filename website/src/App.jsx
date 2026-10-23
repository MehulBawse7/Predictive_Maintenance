import { useEffect, useMemo, useState } from "react";
import {
  deleteRun,
  getConfig,
  getGlobalShap,
  getHistory,
  getLocalShap,
  predictCsv,
  predictOne,
} from "./api.js";

const NAV = [
  ["single", "Single reading"],
  ["csv", "CSV analysis"],
  ["history", "Saved assessments"],
];

function titleCase(value = "") {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function technicianFaultName(output = "") {
  const names = {
    Chiller_Condenser_Fault: "condenser-related issue",
    Chiller_ExpansionValve_Fault: "expansion valve issue",
    Chiller_Evaporator_Fault: "evaporator-related issue",
    Chiller_Compressor_Fault: "compressor-related issue",
    Chiller_General_Fault: "general chiller issue",
    Condenser_Valve_Fault: "condenser valve or filter issue",
    Condenser_Filter_Fault: "condenser valve or filter issue",
    Condenser_GasChargeNeeded: "refrigerant charge issue",
    Condenser_Fan_Fault: "condenser fan issue",
    Condenser_Coil_Fault: "condenser coil or refrigerant leak issue",
    Condenser_GasLeak: "condenser coil or refrigerant leak issue",
    Evaporator_LiquidLeak: "evaporator leak or valve issue",
    Evaporator_Valve_Fault: "evaporator leak or valve issue",
    Evaporator_StrainerFilter_Fault: "evaporator strainer or filter issue",
    Evaporator_Piping_Fault: "evaporator piping issue",
    Evaporator_HeatExchanger_Fault: "evaporator heat exchanger issue",
    Evaporator_Electrical_Fault: "evaporator electrical issue",
    Evaporator_FlowSwitch_Fault: "evaporator flow switch issue",
    Compressor_General_Fault: "compressor issue",
    Compressor_Contactor_Fault: "compressor contactor or oil-loss issue",
    Compressor_Leak: "compressor leak issue",
    Compressor_OilLoss: "compressor contactor or oil-loss issue",
    Compressor_SuperheatIncreaseNeeded: "compressor superheat issue",
    General_Electrical_Fault: "electrical issue",
    General_LowPerformance: "low-performance issue",
    General_UnitUndersizedOversized: "unit sizing issue",
  };
  return names[output] || titleCase(output).toLowerCase();
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || value === "") return "—";
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
}

function localDateTimeValue() {
  const now = new Date();
  return new Date(now.getTime() - now.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
}

function friendlyTime(value) {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? String(value ?? "—") : date.toLocaleString();
}

function App() {
  const [config, setConfig] = useState(null);
  const [tab, setTab] = useState("single");
  const [singleResult, setSingleResult] = useState(null);
  const [csvResults, setCsvResults] = useState(null);
  const [csvIndex, setCsvIndex] = useState(0);
  const [history, setHistory] = useState([]);
  const [historySelection, setHistorySelection] = useState(null);
  const [globalError, setGlobalError] = useState("");

  useEffect(() => {
    getConfig()
      .then(setConfig)
      .catch((error) => setGlobalError(error.message));
    refreshHistory();
  }, []);

  async function refreshHistory() {
    try {
      const response = await getHistory();
      setHistory(response.results || []);
    } catch {
      // Health/config errors are presented in the main status banner.
    }
  }

  const activeResult =
    tab === "csv" && csvResults
      ? csvResults.results[csvIndex]
      : tab === "history"
        ? historySelection
        : singleResult;

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#home" onClick={() => setTab("single")}>
          <span className="brand-mark" aria-hidden="true"><span /></span>
          <span><strong>MachinaSense</strong><small>CHILLER HEALTH SCREENING</small></span>
        </a>
        <div className="topbar-right">
          <span className="prototype-pill"><span className="status-dot" /> Prototype</span>
          <span className="threshold-pill">Attention cutoff <b>&gt; {fmt(config?.attention_threshold ?? 0.5, 2)}</b></span>
        </div>
      </header>

      <main className="page-wrap">
        <section className="intro-row">
          <div>
            <p className="eyebrow">INDUSTRIAL MAINTENANCE SUPPORT</p>
            <h1>Chiller fault screening</h1>
            <p className="intro-copy">Review sensor readings, prioritize attention candidates, and explore what influenced a selected model score.</p>
          </div>
          <div className="model-chip"><span className="chip-icon">M</span><span><b>MLP model</b><small>17 inputs · 26 outputs</small></span></div>
        </section>

        <div className="disclaimer" role="note">
          <span className="disclaimer-icon">i</span>
          <p><b>Decision support only.</b> Scores align with synthetic fuzzy expert-system targets; they are not calibrated real-world failure probabilities or confirmed diagnoses.</p>
        </div>

        {globalError && <div className="error-banner" role="alert"><b>Backend connection issue</b><span>{globalError}</span><small>Start the FastAPI service, then refresh this page.</small></div>}

        <nav className="main-tabs" aria-label="Assessment views">
          {NAV.map(([key, label]) => (
            <button key={key} className={tab === key ? "nav-tab active" : "nav-tab"} onClick={() => setTab(key)}>
              {label}{key === "history" && history.length > 0 && <span className="tab-count">{history.length}</span>}
            </button>
          ))}
        </nav>

        {tab === "single" && <SingleView config={config} onResult={(result) => { setSingleResult(result); refreshHistory(); }} />}
        {tab === "csv" && <CsvView config={config} onResults={(results) => { setCsvResults(results); setCsvIndex(0); refreshHistory(); }} results={csvResults} selectedIndex={csvIndex} setSelectedIndex={setCsvIndex} />}
        {tab === "history" && <HistoryView history={history} selected={historySelection} onSelect={setHistorySelection} onRefresh={refreshHistory} />}

        {activeResult && <ResultDetails result={activeResult} outputColumns={config?.output_columns || []} inputSchema={config?.input_columns || []} />}

        <footer className="page-footer">
          <span>MachinaSense · College project prototype</span>
          <span>Timestamped rows are scored independently; no fault forecast or remaining-useful-life estimate is produced.</span>
        </footer>
      </main>
    </div>
  );
}

function SingleView({ config, onResult }) {
  const [values, setValues] = useState({});
  const [timestamp, setTimestamp] = useState(localDateTimeValue());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!config) return;
    const defaults = Object.fromEntries(config.input_columns.map((input) => [input.name, input.default]));
    setValues(defaults);
  }, [config]);

  const mode = Number(values.Mode_1Cool_2Heat ?? 1);
  useEffect(() => {
    if (!config || !Object.keys(values).length) return;
    const inactiveName = mode === 1 ? "H23C" : "H23F";
    const inactiveValue = mode === 1 ? 10 : 6.5;
    setValues((current) => current[inactiveName] === inactiveValue ? current : { ...current, [inactiveName]: inactiveValue });
  }, [mode, config]);

  async function submit(event) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const result = await predictOne({
        timestamp: new Date(timestamp).toISOString(),
        sensor_readings: Object.fromEntries(Object.entries(values).map(([key, value]) => [key, Number(value)])),
      });
      onResult(result);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="assessment-layout">
      <form className="panel input-panel" onSubmit={submit}>
        <div className="panel-heading">
          <div><p className="eyebrow">NEW ASSESSMENT</p><h2>Enter sensor readings</h2></div>
          <span className="step-number">01</span>
        </div>
        <p className="panel-subtitle">Use the values recorded for one chiller reading. The form follows the model’s saved input order.</p>
        <label className="field-label timestamp-field">Reading timestamp
          <input type="datetime-local" value={timestamp} onChange={(event) => setTimestamp(event.target.value)} required />
        </label>
        <div className="sensor-grid">
          {(config?.input_columns || []).map((input) => {
            const inactive = (mode === 1 && input.name === "H23C") || (mode === 2 && input.name === "H23F");
            return <SensorField key={input.name} input={input} value={values[input.name]} disabled={inactive} onChange={(value) => setValues((current) => ({ ...current, [input.name]: value }))} />;
          })}
        </div>
        <div className="mode-note"><b>Operating mode:</b> H23F is used in Cool mode; H23C is used in Heat mode. The inactive reading is set using the dataset’s midpoint convention. No measurement units are specified in the supplied dictionary.</div>
        {error && <div className="inline-error" role="alert">{error}</div>}
        <button className="primary-button" type="submit" disabled={busy || !config}>{busy ? <><span className="spinner" /> Running prediction…</> : <>Run chiller screening <span>→</span></>}</button>
      </form>
      <aside className="side-guide">
        <div className="panel guide-card">
          <div className="guide-icon">✓</div><h3>What you’ll get</h3>
          <ul><li>26 model scores in the saved output order</li><li>Attention candidates ranked for review</li><li>Global and local SHAP explanations on request</li><li>A saved assessment for later review</li></ul>
        </div>
        <div className="panel guide-card range-card">
          <p className="eyebrow">INPUT RANGE GUIDE</p><h3>Typical vs. accepted</h3>
          <p>Typical ranges are shown beside each field. Values within the wider dataset sampling range are accepted, with a note when outside the typical range.</p>
          <span className="range-key"><i className="key-typical" /> Typical operating range</span>
          <span className="range-key"><i className="key-accepted" /> Accepted dataset range</span>
        </div>
      </aside>
    </section>
  );
}

function SensorField({ input, value, disabled, onChange }) {
  const helper = `${input.normal_min}–${input.normal_max} typical · ${input.min}–${input.max} accepted`;
  if (input.kind === "mode") {
    return <label className="field-label">Operating mode
      <select value={value ?? 1} onChange={(event) => onChange(Number(event.target.value))}><option value="1">Cool (1)</option><option value="2">Heat (2)</option></select>
      <small>Dataset mode selector</small>
    </label>;
  }
  if (input.kind === "binary") {
    return <label className="field-label">{input.name}
      <select value={value ?? 0} onChange={(event) => onChange(Number(event.target.value))}><option value="0">0</option><option value="1">1</option></select>
      <small>Binary input · choose 0 or 1</small>
    </label>;
  }
  return <label className={disabled ? "field-label disabled-field" : "field-label"}>{input.name}
    <input type="number" value={value ?? ""} min={input.min} max={input.max} step="any" required disabled={disabled} onChange={(event) => onChange(event.target.value)} />
    <small>{disabled ? `Inactive in this mode · set to ${value}` : helper}</small>
  </label>;
}

function CsvView({ config, onResults, results, selectedIndex, setSelectedIndex }) {
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [trendOutput, setTrendOutput] = useState("");

  useEffect(() => {
    if (config?.output_columns?.length && !trendOutput) setTrendOutput(config.output_columns[0]);
  }, [config, trendOutput]);

  async function submit(event) {
    event.preventDefault();
    if (!file) return setError("Choose a CSV file first.");
    setBusy(true);
    setError("");
    try {
      const response = await predictCsv(file);
      onResults(response);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  const rows = results?.results || [];
  const complete = rows.filter((row) => row.status === "complete").length;
  const attention = rows.filter((row) => row.attention_candidates?.length).length;
  const displayed = rows.slice(0, 150);

  return <section className="csv-page">
    <form className="panel upload-panel" onSubmit={submit}>
      <div className="upload-copy"><p className="eyebrow">BATCH REVIEW</p><h2>Analyze a timestamped CSV</h2><p>Upload up to {config?.max_csv_rows?.toLocaleString() || "5,000"} readings. Each row is scored independently; missing or invalid sensor values are not imputed.</p></div>
      <label className="dropzone"><input type="file" accept=".csv,text/csv" onChange={(event) => setFile(event.target.files?.[0] || null)} /><span className="upload-icon">↑</span><b>{file?.name || "Choose a CSV file"}</b><small>Required: timestamp plus all 17 model input columns</small></label>
      <button className="primary-button" type="submit" disabled={busy || !config}>{busy ? <><span className="spinner" /> Analyzing rows…</> : <>Analyze CSV <span>→</span></>}</button>
      {error && <div className="inline-error" role="alert">{error}</div>}
      <details className="format-details"><summary>CSV format and column names</summary><p>The backend matches sensor columns by the exact names in the model metadata and reorders them automatically.</p><code>timestamp, {(config?.input_columns || []).map((item) => item.name).join(", ")}</code><p>Mode must be 1 (Cool) or 2 (Heat); H62 must be 0 or 1. Inactive H23F/H23C values are replaced with the dataset midpoint convention.</p></details>
    </form>

    {rows.length > 0 && <>
      <div className="metric-row">
        <Metric label="Rows analyzed" value={rows.length.toLocaleString()} />
        <Metric label="Complete assessments" value={complete.toLocaleString()} />
        <Metric label="Rows with attention candidates" value={attention.toLocaleString()} />
        <Metric label="Threshold" value={`> ${fmt(config?.attention_threshold, 2)}`} />
      </div>
      <div className="panel trend-panel">
        <div className="panel-heading"><div><p className="eyebrow">SCORE OVER TIME</p><h2>Review one output’s trend</h2></div><select className="compact-select" value={trendOutput} onChange={(event) => setTrendOutput(event.target.value)}>{(config?.output_columns || []).map((name) => <option key={name} value={name}>{titleCase(name)}</option>)}</select></div>
        <p className="panel-subtitle">The dashed line is the attention cutoff. This chart displays submitted readings; it is not a forecast.</p>
        <TrendChart rows={rows} outputName={trendOutput} threshold={config?.attention_threshold ?? 0.5} />
      </div>
      <div className="panel table-panel">
        <div className="panel-heading"><div><p className="eyebrow">ROW RESULTS</p><h2>Timestamped assessments</h2></div><span className="muted">Select a row to inspect its scores and SHAP</span></div>
        <div className="table-scroll"><table><thead><tr><th>Timestamp</th><th>Status</th><th>Top attention candidate</th><th>Score</th><th /></tr></thead><tbody>{displayed.map((row, index) => { const candidate = row.attention_candidates?.[0]; return <tr key={`${row.timestamp}-${index}`} className={selectedIndex === index ? "selected-row" : ""}><td>{friendlyTime(row.timestamp)}</td><td><StatusBadge status={row.status} /></td><td>{candidate?.meaning || (row.errors?.length ? "Assessment incomplete" : "No candidate above cutoff")}</td><td>{candidate ? fmt(candidate.model_score) : "—"}</td><td><button className="text-button" onClick={() => setSelectedIndex(index)}>View</button></td></tr>; })}</tbody></table></div>
        {rows.length > displayed.length && <p className="table-note">Showing the first {displayed.length} rows here. All {rows.length.toLocaleString()} predictions are saved and included in the trend.</p>}
      </div>
    </>}
  </section>;
}

function Metric({ label, value }) { return <div className="metric-card"><small>{label}</small><strong>{value}</strong></div>; }

function StatusBadge({ status }) { return <span className={`status-badge ${status === "complete" ? "good" : "incomplete"}`}>{status === "complete" ? "Complete" : "Incomplete"}</span>; }

function TrendChart({ rows, outputName, threshold }) {
  const width = 1000, height = 280, pad = { left: 52, right: 22, top: 20, bottom: 35 };
  const scores = rows.map((row) => row.model_scores?.[outputName]).filter((value) => value !== null && value !== undefined && Number.isFinite(Number(value))).map(Number);
  if (!scores.length) return <div className="empty-chart">No score values available for this output.</div>;
  const minimum = Math.min(0, ...scores), maximum = Math.max(1, ...scores);
  const x = (index) => pad.left + (rows.length <= 1 ? 0 : index * (width - pad.left - pad.right) / (rows.length - 1));
  const y = (value) => pad.top + (maximum - value) * (height - pad.top - pad.bottom) / (maximum - minimum || 1);
  let pathStarted = false;
  const points = rows.map((row, index) => {
    const raw = row.model_scores?.[outputName];
    if (raw === null || raw === undefined || !Number.isFinite(Number(raw))) { pathStarted = false; return ""; }
    const command = pathStarted ? "L" : "M";
    pathStarted = true;
    return `${command}${x(index)},${y(Number(raw))}`;
  }).filter(Boolean).join(" ");
  return <div className="chart-scroll"><svg className="trend-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${titleCase(outputName)} model scores across uploaded timestamps`}>
    {[0, 0.5, 1].map((value) => <g key={value}><line x1={pad.left} x2={width - pad.right} y1={y(value)} y2={y(value)} className={value === threshold ? "threshold-grid" : "chart-grid"} /><text x={pad.left - 10} y={y(value) + 4} textAnchor="end" className="axis-label">{value.toFixed(1)}</text></g>)}
    <line x1={pad.left} x2={width - pad.right} y1={y(threshold)} y2={y(threshold)} className="threshold-line" />
    <path d={points} className="trend-line" />
    {rows.length <= 120 && rows.map((row, index) => { const value = row.model_scores?.[outputName]; return value === null || value === undefined || !Number.isFinite(Number(value)) ? null : <circle key={`${index}`} cx={x(index)} cy={y(Number(value))} r="3" className={row.status === "complete" ? "trend-dot" : "trend-dot invalid-dot"}><title>{friendlyTime(row.timestamp)} · {fmt(value)} · {row.status}</title></circle>; })}
    <text x={pad.left} y={height - 8} className="axis-label">Earlier rows</text><text x={width - pad.right} y={height - 8} textAnchor="end" className="axis-label">Later rows</text>
    <text x={width - pad.right} y={y(threshold) - 8} textAnchor="end" className="threshold-label">Attention cutoff {threshold.toFixed(2)}</text>
  </svg></div>;
}

function HistoryView({ history, selected, onSelect, onRefresh }) {
  const [error, setError] = useState("");
  async function removeRun(runId) {
    if (!window.confirm("Delete all saved rows from this assessment?")) return;
    try {
      await deleteRun(runId);
      onSelect(null);
      onRefresh();
    } catch (requestError) { setError(requestError.message); }
  }
  return <section className="panel history-panel">
    <div className="panel-heading"><div><p className="eyebrow">LOCAL DATABASE</p><h2>Saved assessments</h2></div><button className="secondary-button" onClick={onRefresh}>Refresh</button></div>
    <p className="panel-subtitle">Recent single readings and CSV rows saved on this server in SQLite.</p>
    {error && <div className="inline-error">{error}</div>}
    {!history.length ? <div className="empty-state"><span>▤</span><b>No saved assessments yet</b><small>Run a single prediction or CSV analysis to create history.</small></div> : <div className="history-list">{history.map((item) => <div className={`history-item ${selected?.id === item.id ? "history-active" : ""}`} key={item.id}>
      <button className="history-open" onClick={() => onSelect(item)}><span className="history-source">{item.source === "csv" ? "CSV row" : "Single reading"} · Run {item.run_id}</span><b>{friendlyTime(item.timestamp)}</b><span>{item.attention_candidates?.[0]?.meaning || (item.status === "incomplete" ? "Assessment incomplete" : "No candidates above cutoff")}</span></button>
      <StatusBadge status={item.status} /><button className="delete-button" onClick={() => removeRun(item.run_id)} aria-label={`Delete run ${item.run_id}`} title="Delete assessment">×</button>
    </div>)}</div>}
  </section>;
}

function ResultDetails({ result, outputColumns, inputSchema }) {
  const candidates = result.attention_candidates || [];
  const scoreEntries = useMemo(() => Object.entries(result.model_scores || {}).sort((a, b) => {
    const scoreA = a[1] === null || a[1] === undefined || !Number.isFinite(Number(a[1])) ? -Infinity : Number(a[1]);
    const scoreB = b[1] === null || b[1] === undefined || !Number.isFinite(Number(b[1])) ? -Infinity : Number(b[1]);
    return scoreB - scoreA;
  }), [result]);
  return <section className="result-section" id="assessment-results">
    <div className="results-heading"><div><p className="eyebrow">ASSESSMENT RESULTS</p><h2>Reading review</h2></div><div className="result-time">{friendlyTime(result.timestamp)}<small className="model-version">Model {result.model_version || "saved MLP"}</small></div></div>
    {result.status === "incomplete" ? <div className="incomplete-banner"><span className="warning-symbol">!</span><div><b>Assessment incomplete — candidates were not ranked</b><p>One or more model scores were outside the expected range. This is a prediction-range issue; entering the same readings again may return the same scores.</p><ul>{(result.errors || []).map((error) => <li key={error}>{error}</li>)}</ul></div></div> : <div className="result-summary"><div><StatusBadge status={result.status} /><h3>{candidates.length ? `${candidates.length} attention ${candidates.length === 1 ? "candidate" : "candidates"}` : "No candidates passed the attention rule"}</h3><p>Attention rule: score strictly greater than {fmt(result.threshold, 2)}. Scores are not failure probabilities.</p></div><div className="result-stat"><small>TOP SCORE</small><strong>{candidates.length ? fmt(candidates[0].model_score) : "—"}</strong><small>{candidates[0]?.meaning || "No attention candidate"}</small></div></div>}
    {!!result.input_warnings?.length && <div className="warning-list"><b>Outside typical range</b><span>Accepted by the dataset sampling limits, but these readings are outside the dictionary’s typical range:</span><ul>{result.input_warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></div>}
    <div className="panel candidate-panel">
      <div className="panel-heading"><div><p className="eyebrow">PRIORITIZED REVIEW</p><h2>Attention candidates</h2></div><span className="muted">Ranked by model score</span></div>
      {result.status === "incomplete" ? <p className="empty-inline">No ranking is shown because the score validation failed for this row.</p> : candidates.length ? <div className="candidate-list">{candidates.map((item) => <div className="candidate-row" key={item.outputs.join("+")}><span className="candidate-rank">{String(item.rank).padStart(2, "0")}</span><div className="candidate-copy"><b>{item.meaning || item.outputs.map(titleCase).join(" / ")}</b><small>{item.interpretation}</small>{item.parent_context && <small>{item.parent_context}</small>}<small className="action-copy">Suggested next step: inspect the indicated subsystem; a qualified technician should determine whether repair is needed.</small></div><div className="candidate-score"><strong>{fmt(item.model_score)}</strong><small>model score</small></div></div>)}</div> : <div className="empty-inline">No output exceeded the selected cutoff for this reading.</div>}
      <p className="candidate-footnote">An attention candidate is not a confirmed fault. Inspection by qualified maintenance staff is recommended.</p>
    </div>
    <details className="panel sensor-evidence"><summary><span><b>Sensor evidence · 17 readings</b><small>Measurements submitted for this timestamp</small></span><span className="disclosure">＋</span></summary><div className="evidence-grid">{inputSchema.map((input) => { const value = result.sensor_readings?.[input.name]; const outOfTypical = value !== null && value !== undefined && input.kind !== "mode" && !(Number(value) >= input.normal_min && Number(value) <= input.normal_max); return <div className={outOfTypical ? "evidence-item evidence-warning" : "evidence-item"} key={input.name}><span>{input.name}</span><b>{fmt(value, 3)}</b>{outOfTypical && <small>Outside typical {input.normal_min}–{input.normal_max}</small>}</div>; })}</div></details>
    <details className="panel all-scores-panel"><summary><span><b>All 26 model scores</b><small>Scores outside [0, 1] are marked invalid.</small></span><span className="disclosure">＋</span></summary><div className="score-grid">{scoreEntries.map(([name, score]) => { const valid = Number.isFinite(Number(score)) && Number(score) >= 0 && Number(score) <= 1; const candidate = candidates.some((item) => item.outputs.includes(name)); return <div className={`score-item ${!valid ? "score-invalid" : ""}`} key={name}><span>{titleCase(name)}{candidate && <i className="mini-alert">attention</i>}</span><b>{fmt(score)}</b></div>; })}</div></details>
    {!!outputColumns.length && <ExplanationPanel result={result} outputColumns={outputColumns} />}
  </section>;
}

function ExplanationPanel({ result, outputColumns }) {
  const bestOutput = result.attention_candidates?.[0]?.outputs?.[0] || Object.entries(result.model_scores || {}).sort((a, b) => Number(b[1]) - Number(a[1]))[0]?.[0] || outputColumns[0];
  const [output, setOutput] = useState(bestOutput);
  const [global, setGlobal] = useState(null);
  const [local, setLocal] = useState(null);
  const [loading, setLoading] = useState("");
  const [error, setError] = useState("");
  const selectedScore = Number(result.model_scores?.[output]);
  const selectedScoreValid = Number.isFinite(selectedScore) && selectedScore >= 0 && selectedScore <= 1;
  useEffect(() => { setOutput(bestOutput); setGlobal(null); setLocal(null); setError(""); }, [result, bestOutput]);

  async function loadGlobal() {
    setLoading("global"); setError("");
    try { setGlobal(await getGlobalShap(output)); }
    catch (requestError) { setError(requestError.message); }
    finally { setLoading(""); }
  }
  async function loadLocal() {
    setLoading("local"); setError("");
    try { setLocal(await getLocalShap({ output_name: output, sensor_readings: result.sensor_readings })); }
    catch (requestError) { setError(requestError.message); }
    finally { setLoading(""); }
  }

  return <div className="panel explanation-panel">
    <div className="panel-heading"><div><p className="eyebrow">TECHNICIAN SUMMARY</p><h2>What to review</h2></div></div>
    {result.status !== "complete" ? <p className="panel-subtitle">This reading could not be ranked. Review the input readings and model output before drawing conclusions.</p> : result.attention_candidates?.length ? <><p className="panel-subtitle">The model flagged these areas for inspection. These are screening results, not confirmed faults or failure probabilities.</p><ul className="technician-summary-list">{result.attention_candidates.slice(0, 5).map((item) => <li key={item.outputs.join("+")}><b>Possible {technicianFaultName(item.outputs[0])}</b> — score {fmt(item.model_score)}. Inspect the indicated area and have a qualified technician assess it.</li>)}</ul></> : <p className="panel-subtitle">No model output crossed the attention cutoff for this reading. This does not guarantee the chiller is fault-free.</p>}
    <details className="technical-explanation">
      <summary>Technical model explanation (SHAP)</summary>
      <p className="panel-subtitle">SHAP shows which inputs moved a selected model score up or down. It does not establish a physical cause. Negative contributions lower that score; they are not negative fault probabilities.</p>
      <div className="explanation-controls"><label className="field-label">Selected output<select value={output} onChange={(event) => { setOutput(event.target.value); setGlobal(null); setLocal(null); }} aria-label="Select one model output">{outputColumns.map((name) => <option key={name} value={name}>{titleCase(name)}</option>)}</select></label><div className="explanation-buttons"><button className="secondary-button" onClick={loadGlobal} disabled={!!loading}>{loading === "global" ? "Loading global SHAP…" : "Global importance + beeswarm"}</button><button className="primary-button compact-button" onClick={loadLocal} disabled={!!loading || !selectedScoreValid}>{loading === "local" ? <><span className="spinner" /> Calculating…</> : "Explain this reading"}</button></div></div>
      {!selectedScoreValid && <p className="field-help-error">Local SHAP is unavailable for this output because its score is outside the valid [0, 1] range. Other outputs with valid scores can still be explained.</p>}
      {error && <div className="inline-error" role="alert">{error}</div>}
      {global && <GlobalShapCharts data={global} />}
      {local && <LocalWaterfall data={local} />}
    </details>
  </div>;
}

function GlobalShapCharts({ data }) {
  const top = data.importance;
  return <div className="shap-charts">
    <div className="chart-heading"><b>Global feature importance · all 17 inputs</b><span>Mean absolute SHAP · {data.explanation_rows} explanation rows</span></div>
    <div className="importance-chart">{top.map((item, index) => { const max = top[0]?.mean_abs_shap || 1; return <div className="importance-row" key={item.feature}><span>{item.feature}</span><div className="importance-track"><i style={{ width: `${100 * item.mean_abs_shap / max}%` }} /></div><b>{fmt(item.mean_abs_shap, 4)}</b><small>{index + 1}</small></div>; })}</div>
    <div className="chart-heading beeswarm-heading"><b>SHAP summary / beeswarm</b><span>Each dot is one sampled row · color shows feature value</span></div>
    <Beeswarm data={data} features={top.map((item) => item.feature)} />
    <div className="color-legend"><span>Low feature value</span><i /><span>High feature value</span><span className="axis-caption">SHAP contribution (negative lowers this score · positive raises it)</span></div>
  </div>;
}

function Beeswarm({ data, features }) {
  const width = 1000, rowHeight = 36, pad = { left: 190, right: 25, top: 15, bottom: 34 };
  const height = pad.top + features.length * rowHeight + pad.bottom;
  const featureIndices = features.map((name) => data.feature_names.indexOf(name));
  const allValues = featureIndices.flatMap((column) => data.shap_values.map((row) => Number(row[column]))).filter(Number.isFinite);
  const extent = Math.max(0.001, ...allValues.map(Math.abs));
  const x = (value) => pad.left + (value + extent) * (width - pad.left - pad.right) / (2 * extent);
  const points = [];
  features.forEach((feature, rowIndex) => {
    const column = featureIndices[rowIndex];
    const featureVals = data.feature_values.map((row) => Number(row[column]));
    const min = Math.min(...featureVals), max = Math.max(...featureVals);
    data.shap_values.forEach((row, sampleIndex) => {
      const rawShap = Number(row[column]);
      const rawValue = featureVals[sampleIndex];
      if (!Number.isFinite(rawShap) || !Number.isFinite(rawValue)) return;
      const colorRatio = max === min ? 0.5 : (rawValue - min) / (max - min);
      const jitter = ((sampleIndex * 7 + rowIndex * 3) % 13 - 6) * 1.45;
      points.push(<circle key={`${rowIndex}-${sampleIndex}`} cx={x(rawShap)} cy={pad.top + rowIndex * rowHeight + rowHeight / 2 + jitter} r="2.1" fill={blendColor(colorRatio)} opacity="0.67"><title>{`${feature}: value ${fmt(rawValue, 3)}, SHAP ${fmt(rawShap, 4)}`}</title></circle>);
    });
  });
  return <div className="chart-scroll"><svg viewBox={`0 0 ${width} ${height}`} className="beeswarm-svg" role="img" aria-label="SHAP beeswarm summary plot">
    {[-1, -0.5, 0, 0.5, 1].map((tick) => { const value = tick * extent; return <g key={tick}><line x1={x(value)} x2={x(value)} y1={pad.top} y2={height - pad.bottom} className={tick === 0 ? "zero-line" : "chart-grid"} /><text x={x(value)} y={height - 9} textAnchor="middle" className="axis-label">{fmt(value, 3)}</text></g>; })}
    {features.map((feature, index) => <g key={feature}><text x={pad.left - 12} y={pad.top + index * rowHeight + rowHeight / 2 + 4} textAnchor="end" className="beeswarm-feature">{feature}</text><line x1={pad.left} x2={width - pad.right} y1={pad.top + (index + 1) * rowHeight} y2={pad.top + (index + 1) * rowHeight} className="chart-grid" /></g>)}
    {points}
  </svg></div>;
}

function blendColor(ratio) {
  const low = [43, 155, 181], high = [225, 72, 129];
  const color = low.map((value, index) => Math.round(value + (high[index] - value) * Math.max(0, Math.min(1, ratio))));
  return `rgb(${color.join(",")})`;
}

function LocalWaterfall({ data }) {
  const contributions = Object.entries(data.shap_values).map(([feature, value]) => ({ feature, value: Number(value), input: data.feature_values[feature] })).sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  let cumulative = Number(data.baseline);
  const steps = contributions.map((item) => { const before = cumulative; cumulative += item.value; return { ...item, before, after: cumulative }; });
  const all = [Number(data.baseline), Number(data.predicted_score), ...steps.flatMap((step) => [step.before, step.after])];
  const min = Math.min(...all), max = Math.max(...all), span = max - min || 1;
  const width = 1000, rowHeight = 30, pad = { left: 230, right: 40, top: 40, bottom: 38 };
  const height = pad.top + steps.length * rowHeight + pad.bottom;
  const x = (value) => pad.left + (value - min) * (width - pad.left - pad.right) / span;
  const zero = Number(data.baseline);
  return <div className="local-chart-wrap">
    <div className="chart-heading"><b>Local explanation · {titleCase(data.output_name)}</b><span>Baseline {fmt(zero)} → predicted score {fmt(data.predicted_score)}</span></div>
    <div className="chart-scroll"><svg viewBox={`0 0 ${width} ${height}`} className="waterfall-svg" role="img" aria-label="Local SHAP contribution waterfall">
      <line x1={x(zero)} x2={x(zero)} y1={pad.top - 16} y2={height - pad.bottom} className="zero-line" />
      <text x={x(zero)} y={17} textAnchor="middle" className="axis-label">Baseline {fmt(zero)}</text>
      {steps.map((step, index) => { const y = pad.top + index * rowHeight; const start = Math.min(x(step.before), x(step.after)); const barWidth = Math.max(2, Math.abs(x(step.after) - x(step.before))); return <g key={step.feature}><text x={pad.left - 12} y={y + 15} textAnchor="end" className="waterfall-label">{fmt(step.input, 3)} = {step.feature}</text><rect x={start} y={y + 3} width={barWidth} height="18" rx="4" className={step.value >= 0 ? "positive-bar" : "negative-bar"}><title>{`${step.feature}: SHAP ${fmt(step.value, 4)}`}</title></rect><text x={step.after >= step.before ? x(step.after) + 6 : x(step.after) - 6} y={y + 17} textAnchor={step.after >= step.before ? "start" : "end"} className="contribution-label">{step.value >= 0 ? "+" : ""}{fmt(step.value, 3)}</text></g>; })}
      <text x={pad.left} y={height - 10} className="axis-label">Model output scale</text><text x={x(Number(data.predicted_score))} y={height - 10} textAnchor="middle" className="axis-label">Prediction {fmt(data.predicted_score)}</text>
    </svg></div>
    <p className="plot-note">Positive contributions push this selected score above its baseline; negative contributions push it below. This is an explanation of the model, not proof of physical cause.</p>
  </div>;
}

export default App;
