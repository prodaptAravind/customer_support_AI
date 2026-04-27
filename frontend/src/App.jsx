import { useEffect, useMemo, useState } from 'react';

const SAMPLE_COMPLAINTS = [
  { label: 'Damaged item', text: 'My product arrived damaged. Can I get a refund?' },
  { label: 'Late delivery', text: 'The delivery was delayed by a week and I want to cancel.' },
  { label: 'Wrong item', text: 'I received the wrong item in my package.' },
];

const DEFAULT_API_BASE = import.meta.env.VITE_API_BASE_URL ?? '/api';
const DOCS_PER_PAGE = 3;

function App() {
  const [complaint, setComplaint] = useState(SAMPLE_COMPLAINTS[0].text);
  const [mode, setMode] = useState('strict');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking...');
  const [docPage, setDocPage] = useState(1);
  const [includeMetrics, setIncludeMetrics] = useState(false);

  const modeHint = useMemo(() => {
    return mode === 'friendly'
      ? 'Friendly mode keeps the reply warmer and more empathetic.'
      : 'Strict mode keeps the reply concise and policy-bound.';
  }, [mode]);

  const pageState = useMemo(() => {
    const docs = result?.retrieved_docs ?? [];
    const totalPages = Math.max(1, Math.ceil(docs.length / DOCS_PER_PAGE));
    const safePage = Math.min(docPage, totalPages);
    const start = (safePage - 1) * DOCS_PER_PAGE;

    return {
      docs: docs.slice(start, start + DOCS_PER_PAGE),
      totalDocs: docs.length,
      totalPages,
      safePage,
    };
  }, [docPage, result]);

  useEffect(() => {
    const controller = new AbortController();

    fetch(`${DEFAULT_API_BASE}/health`, { signal: controller.signal })
      .then((response) => response.json())
      .then((data) => {
        setBackendStatus(
          `${data.status} | ${data.retrieval_backend} | rerank ${data.reranking_backend} | chunks ${data.chunk_size}/${data.chunk_overlap} | ${data.llm_provider}`
        );
      })
      .catch(() => {
        setBackendStatus('unreachable');
      });

    return () => controller.abort();
  }, []);

  const useSample = (sample) => {
    setComplaint(sample);
    setResult(null);
    setError('');
    setDocPage(1);
  };

  const submit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);
    setDocPage(1);

    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 45000);

    try {
      const response = await fetch(`${DEFAULT_API_BASE}/respond`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          complaint,
          mode,
          top_k: 3,
          include_metrics: includeMetrics,
        }),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Request failed with status ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      if (err.name === 'AbortError') {
        setError('Request timed out after 45 seconds. The backend may be waiting on OpenAI or Pinecone.');
      } else {
        setError(err.message || 'Something went wrong');
      }
    } finally {
      window.clearTimeout(timeoutId);
      setLoading(false);
    }
  };

  return (
    <div className="shell">
      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Support reply generator</p>
          <h1>Generate customer support replies</h1>
          <p className="lede">Use policy retrieval and tone modes to draft a response quickly.</p>
        </div>

        <div className="hero-panel">
          <div className="stat">
            <span>Backend</span>
            <strong>{backendStatus}</strong>
          </div>
          <div className="stat">
            <span>Mode</span>
            <strong>{mode === 'friendly' ? 'Friendly' : 'Strict'}</strong>
          </div>
        </div>
      </header>

      <main className="layout">
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

            <label className="toggle-row">
              <input
                type="checkbox"
                checked={includeMetrics}
                onChange={(e) => setIncludeMetrics(e.target.checked)}
              />
              Include RAGAS metrics
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
                {result.raw_llm_response?.error ? (
                  <div className="alert error">Backend error: {result.raw_llm_response.error}</div>
                ) : null}
                {result.ragas_metrics ? (
                  <div className="metrics-panel">
                    <div className="metric">
                      <span>Faithfulness</span>
                      <strong>{result.ragas_metrics.faithfulness?.toFixed?.(3) ?? 'n/a'}</strong>
                    </div>
                    <div className="metric">
                      <span>Answer relevancy</span>
                      <strong>{result.ragas_metrics.answer_relevancy?.toFixed?.(3) ?? 'n/a'}</strong>
                    </div>
                    <div className="metric">
                      <span>Context precision</span>
                      <strong>{result.ragas_metrics.context_precision?.toFixed?.(3) ?? 'n/a'}</strong>
                    </div>
                    <div className="metric">
                      <span>Context utilization</span>
                      <strong>{result.ragas_metrics.context_utilization?.toFixed?.(3) ?? 'n/a'}</strong>
                    </div>
                    <div className="metric">
                      <span>Evaluator</span>
                      <strong>{result.ragas_metrics.backend}</strong>
                    </div>
                  </div>
                ) : null}
              </>
            ) : (
              <p className="placeholder">Your generated reply will appear here.</p>
            )}
          </div>

          <div className="card docs-card">
            <div className="section-header">
              <h2>Retrieved policy docs</h2>
              <span className="subtle">
                {result ? `${pageState.totalDocs} documents` : 'Waiting for a query'}
              </span>
            </div>

            {pageState.docs.length ? (
              <>
                <div className="doc-list">
                  {pageState.docs.map((doc) => (
                    <article key={doc.id} className="doc">
                      <div className="doc-top">
                        <h3>{doc.title}</h3>
                        <span className="score">{doc.score.toFixed(2)}</span>
                      </div>
                      <p className="subtle">
                        Source {doc.source_id} - chunk {doc.chunk_index}/{doc.chunk_count}
                      </p>
                      <p>{doc.company_response}</p>
                      <div className="doc-foot">
                        <span>{doc.category}</span>
                        <span>{doc.solution}</span>
                      </div>
                    </article>
                  ))}
                </div>

                {pageState.totalPages > 1 ? (
                  <div className="pagination">
                    <button
                      type="button"
                      className="ghost"
                      onClick={() => setDocPage((page) => Math.max(1, page - 1))}
                      disabled={pageState.safePage === 1}
                    >
                      Previous
                    </button>
                    <span className="page-indicator">
                      Page {pageState.safePage} of {pageState.totalPages}
                    </span>
                    <button
                      type="button"
                      className="ghost"
                      onClick={() => setDocPage((page) => Math.min(pageState.totalPages, page + 1))}
                      disabled={pageState.safePage === pageState.totalPages}
                    >
                      Next
                    </button>
                  </div>
                ) : null}
              </>
            ) : (
              <p className="placeholder">Top matches from the local dataset will show up here.</p>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
