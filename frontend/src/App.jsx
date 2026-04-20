import { useMemo, useState } from 'react';

const SAMPLE_COMPLAINTS = [
  { label: 'Damaged item', text: 'My product arrived damaged. Can I get a refund?' },
  { label: 'Late delivery', text: 'The delivery was delayed by a week and I want to cancel.' },
  { label: 'Wrong item', text: 'I received the wrong item in my package.' },
];

const DEFAULT_API_BASE = import.meta.env.VITE_API_BASE_URL ?? '/api';

function App() {
  const [complaint, setComplaint] = useState(SAMPLE_COMPLAINTS[0].text);
  const [mode, setMode] = useState('strict');
  const [temperature, setTemperature] = useState(0.2);
  const [maxTokens, setMaxTokens] = useState(150);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  const modeHint = useMemo(
    () =>
      mode === 'friendly'
        ? 'Friendly mode keeps the answer warm and empathetic.'
        : 'Strict mode keeps the reply concise and policy-bound.',
    [mode]
  );

  const useSample = (sample) => {
    setComplaint(sample);
    setResult(null);
    setError('');
  };

  const submit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await fetch(`${DEFAULT_API_BASE}/respond`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          complaint,
          mode,
          temperature,
          max_tokens: maxTokens,
          top_k: 3,
        }),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Request failed with status ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="shell">
      <div className="hero">
        <div className="hero-copy">
          <p className="eyebrow">BM25 + Sarvam AI</p>
          <h1>AI-Assisted Customer Support Response Generator</h1>
          <p className="lede">
            Draft support replies from local policy documents with controlled prompts,
            tone switching, and a built-in escalation fallback.
          </p>
        </div>
        <div className="hero-panel">
          <div className="stat">
            <span>Retrieval</span>
            <strong>Top 3 policy matches</strong>
          </div>
          <div className="stat">
            <span>Modes</span>
            <strong>Strict or Friendly</strong>
          </div>
          <div className="stat">
            <span>LLM</span>
            <strong>Sarvam chat completions</strong>
          </div>
        </div>
      </div>

      <div className="layout">
        <form className="composer card" onSubmit={submit}>
          <label>
            Customer complaint
            <textarea
              value={complaint}
              onChange={(e) => setComplaint(e.target.value)}
              placeholder="Describe the customer issue here..."
              rows={6}
            />
          </label>

          <div className="sample-row">
            {SAMPLE_COMPLAINTS.map((sample) => (
              <button
                key={sample.label}
                type="button"
                className="ghost"
                onClick={() => useSample(sample.text)}
              >
                {sample.label}
              </button>
            ))}
          </div>

          <div className="controls">
            <label>
              Mode
              <select value={mode} onChange={(e) => setMode(e.target.value)}>
                <option value="strict">Strict policy mode</option>
                <option value="friendly">Friendly tone mode</option>
              </select>
            </label>

            <label>
              Temperature: <strong>{temperature.toFixed(1)}</strong>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
              />
            </label>

            <label>
              Max tokens: <strong>{maxTokens}</strong>
              <input
                type="range"
                min="100"
                max="250"
                step="10"
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
              />
            </label>
          </div>

          <p className="hint">{modeHint}</p>

          <button className="primary" type="submit" disabled={loading || !complaint.trim()}>
            {loading ? 'Generating response...' : 'Generate support reply'}
          </button>

          {error ? <div className="alert error">{error}</div> : null}
        </form>

        <section className="results">
          <div className="card response-card">
            <div className="section-header">
              <h2>Generated response</h2>
              {result ? (
                <span className={`badge ${result.fallback_used ? 'warn' : 'ok'}`}>
                  {result.fallback_used ? 'Escalation fallback' : 'Policy-backed'}
                </span>
              ) : null}
            </div>

            {result ? (
              <>
                <p className="response-text">{result.answer}</p>
                <div className="meta-grid">
                  <div>
                    <span>Mode</span>
                    <strong>{result.mode}</strong>
                  </div>
                  <div>
                    <span>Temperature</span>
                    <strong>{result.temperature}</strong>
                  </div>
                  <div>
                    <span>Max tokens</span>
                    <strong>{result.max_tokens}</strong>
                  </div>
                  <div>
                    <span>LLM provider</span>
                    <strong>{result.llm_provider}</strong>
                  </div>
                </div>
              </>
            ) : (
              <p className="placeholder">
                Your generated reply will appear here, along with the prompt mode used.
              </p>
            )}
          </div>

          <div className="card docs-card">
            <div className="section-header">
              <h2>Retrieved policy docs</h2>
              <span className="subtle">
                {result ? `${result.retrieved_docs.length} documents` : 'Waiting for a query'}
              </span>
            </div>

            {result?.retrieved_docs?.length ? (
              <div className="doc-list">
                {result.retrieved_docs.map((doc) => (
                  <article key={doc.id} className="doc">
                    <div className="doc-top">
                      <h3>{doc.title}</h3>
                      <span className="score">{doc.score.toFixed(2)}</span>
                    </div>
                    <p>{doc.company_response}</p>
                    <div className="doc-foot">
                      <span>{doc.category}</span>
                      <span>{doc.solution}</span>
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <p className="placeholder">Top matches from the local dataset will show up here.</p>
            )}
          </div>

          <div className="card prompt-card">
            <div className="section-header">
              <h2>Prompt preview</h2>
              <span className="subtle">Useful for the assignment write-up</span>
            </div>
            {result ? (
              <div className="prompt-box">
                <strong>System</strong>
                <pre>{result.prompt.system}</pre>
                <strong>User</strong>
                <pre>{result.prompt.user}</pre>
              </div>
            ) : (
              <p className="placeholder">The exact prompt used for generation will be shown here.</p>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

export default App;
