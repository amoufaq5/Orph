import React, { useEffect, useRef, useState } from "react";

/**
 * Props:
 * - studioUrl: string (e.g., http://localhost:5055)
 */
export default function BuildPanel({ studioUrl = "http://localhost:5055" }) {
  const [userId, setUserId] = useState("demo");
  const [seed, setSeed] = useState(42);
  const [trainRatio, setTrainRatio] = useState(0.94);
  const [valRatio, setValRatio] = useState(0.03);
  const [testRatio, setTestRatio] = useState(0.03);

  const [jobId, setJobId] = useState("");
  const [status, setStatus] = useState("");
  const [logs, setLogs] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const logRef = useRef(null);
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  async function startBuild() {
    setError(""); setBusy(true); setJobId(""); setStatus(""); setLogs("");
    try {
      const body = {
        user_id: userId,
        seed: Number(seed),
        train_ratio: Number(trainRatio),
        val_ratio: Number(valRatio),
        test_ratio: Number(testRatio)
      };
      const res = await fetch(`${studioUrl}/build_dataset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok || !data?.job_id) throw new Error(data?.message || "Failed to start build");
      setJobId(data.job_id);
      setStatus("queued");
    } catch (e) {
      setError(e.message || String(e));
      setBusy(false);
    }
  }

  // Poll status/logs
  useEffect(() => {
    if (!jobId) return;
    let alive = true;
    setBusy(true);

    async function poll() {
      try {
        const [stRes, lgRes] = await Promise.all([
          fetch(`${studioUrl}/status/${jobId}`),
          fetch(`${studioUrl}/logs/${jobId}`)
        ]);
        const st = await stRes.json();
        const lg = await lgRes.json();
        if (!alive) return;
        setStatus(st?.status || "unknown");
        setLogs(lg?.logs || "");
        if (["completed","failed","unknown"].includes(st?.status)) {
          setBusy(false); return;
        }
        setTimeout(poll, 1500);
      } catch (e) {
        if (!alive) return;
        setError(e.message || String(e));
        setBusy(false);
      }
    }
    poll();
    return () => { alive = false; };
  }, [jobId, studioUrl]);

  const reportUrl = jobId ? `${studioUrl}/build_report/${jobId}` : "";
  const downloadUrl = jobId ? `${studioUrl}/download_model/${jobId}` : "";

  return (
    <div style={styles.wrap}>
      <h3 style={styles.h3}>Build Unified Dataset</h3>
      <div style={styles.row}>
        <label style={styles.label}>Studio URL</label>
        <input style={styles.input} value={studioUrl} disabled />
      </div>
      <div style={styles.grid}>
        <div>
          <label style={styles.label}>User ID</label>
          <input style={styles.input} value={userId} onChange={e=>setUserId(e.target.value)} disabled={busy}/>
        </div>
        <div>
          <label style={styles.label}>Seed</label>
          <input type="number" style={styles.input} value={seed} onChange={e=>setSeed(e.target.value)} disabled={busy}/>
        </div>
        <div>
          <label style={styles.label}>Train</label>
          <input type="number" step="0.01" style={styles.input} value={trainRatio} onChange={e=>setTrainRatio(e.target.value)} disabled={busy}/>
        </div>
        <div>
          <label style={styles.label}>Val</label>
          <input type="number" step="0.01" style={styles.input} value={valRatio} onChange={e=>setValRatio(e.target.value)} disabled={busy}/>
        </div>
        <div>
          <label style={styles.label}>Test</label>
          <input type="number" step="0.01" style={styles.input} value={testRatio} onChange={e=>setTestRatio(e.target.value)} disabled={busy}/>
        </div>
      </div>

      <div style={styles.btnRow}>
        <button onClick={startBuild} disabled={busy} style={busy ? styles.btnDisabled : styles.btn}>
          {busy && !jobId ? "Starting..." : "Start Build"}
        </button>
        {jobId ? <span style={styles.meta}>Job: <code>{jobId}</code> • Status: <b>{status}</b></span> : null}
        {status === "completed" ? (
          <>
            <a style={styles.linkBtn} href={reportUrl} target="_blank" rel="noreferrer">Open Build Report</a>
            <a style={styles.linkBtn} href={downloadUrl}>Download Artifacts (.zip)</a>
          </>
        ) : null}
      </div>

      {error ? <div style={styles.error}>⚠ {error}</div> : null}
      <div style={styles.logs} ref={logRef}>
        <pre style={styles.pre}>{logs || (jobId ? "Waiting for logs..." : "Logs will appear here…")}</pre>
      </div>
    </div>
  );
}

const styles = {
  wrap: { maxWidth: 920, margin: "16px auto", padding: 16, border: "1px solid #e5e7eb", borderRadius: 8, fontFamily: "Inter, system-ui, sans-serif", background: "#fff" },
  h3: { marginTop: 0, marginBottom: 12 },
  row: { marginBottom: 10 },
  label: { display: "block", fontSize: 13, color: "#374151", marginBottom: 4 },
  input: { width: "100%", padding: 8, border: "1px solid #d1d5db", borderRadius: 6, outline: "none" },
  grid: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 10, marginBottom: 8 },
  btnRow: { display: "flex", alignItems: "center", gap: 10, marginTop: 10, flexWrap: "wrap" },
  btn: { padding: "8px 12px", background: "#111827", color: "white", borderRadius: 6, border: "none", cursor: "pointer" },
  btnDisabled: { padding: "8px 12px", background: "#9ca3af", color: "white", borderRadius: 6, border: "none", cursor: "not-allowed" },
  linkBtn: { padding: "6px 10px", background: "#065f46", color: "white", borderRadius: 6, textDecoration: "none" },
  meta: { fontSize: 12, color: "#374151" },
  error: { color: "#b91c1c", fontSize: 13, marginTop: 6 },
  logs: { marginTop: 10, border: "1px solid #e5e7eb", borderRadius: 6, background: "#f9fafb", minHeight: 160, maxHeight: 360, overflowY: "auto" },
  pre: { margin: 0, padding: 8, whiteSpace: "pre-wrap", fontFamily: "ui-monospace, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace" },
};
