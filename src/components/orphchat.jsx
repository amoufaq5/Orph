import React, { useEffect, useMemo, useRef, useState } from "react";

/**
 * Props:
 * - apiUrl: string  -> Inference base URL (e.g., http://localhost:6060)
 * - mode: "sft" | "cot"
 * - maxNewTokens: number
 * - title?: string
 */
export default function OrphChat({
  apiUrl = "http://localhost:6060",
  mode = "sft",
  maxNewTokens = 256,
  title = "Orph Inference",
}) {
  const [messages, setMessages] = useState([
    { role: "system", text: "You’re chatting with Orph. This demo streams tokens." },
  ]);
  const [instruction, setInstruction] = useState("");
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const scrollRef = useRef(null);

  const canSend = useMemo(() => instruction.trim().length > 0 && !busy, [instruction, busy]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, busy]);

  async function sendStream() {
    setError("");
    setBusy(true);
    const userMsg = { role: "user", text: `Instruction:\n${instruction}\n\nInput:\n${input}` };
    setMessages((m) => [...m, userMsg, { role: "assistant", text: "" }]);

    try {
      const res = await fetch(`${apiUrl}/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          instruction,
          input,
          mode,
          max_new_tokens: maxNewTokens,
          temperature: 0.7,
          top_p: 0.9,
        }),
      });

      if (!res.ok || !res.body) {
        const txt = await res.text();
        throw new Error(txt || `HTTP ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let acc = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        acc += chunk;
        setMessages((m) => {
          const copy = [...m];
          // last message is assistant placeholder
          copy[copy.length - 1] = { role: "assistant", text: acc };
          return copy;
        });
      }
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  async function sendOnce() {
    setError("");
    setBusy(true);
    const userMsg = { role: "user", text: `Instruction:\n${instruction}\n\nInput:\n${input}` };
    setMessages((m) => [...m, userMsg]);

    try {
      const res = await fetch(`${apiUrl}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          instruction,
          input,
          mode,
          max_new_tokens: maxNewTokens,
          temperature: 0.7,
          top_p: 0.9,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || `HTTP ${res.status}`);
      setMessages((m) => [...m, { role: "assistant", text: data.completion || "" }]);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={styles.shell}>
      <h3 style={styles.header}>{title}</h3>

      <div style={styles.formRow}>
        <label style={styles.label}>Instruction</label>
        <textarea
          style={styles.textarea}
          placeholder="e.g., Advise a patient with mild tension headache."
          value={instruction}
          onChange={(e) => setInstruction(e.target.value)}
          disabled={busy}
        />
      </div>

      <div style={styles.formRow}>
        <label style={styles.label}>Input (optional)</label>
        <textarea
          style={styles.textarea}
          placeholder="e.g., Age 28, no red flags."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={busy}
        />
      </div>

      <div style={styles.btnRow}>
        <button style={canSend ? styles.btn : styles.btnDisabled} onClick={sendStream} disabled={!canSend}>
          Stream
        </button>
        <button style={canSend ? styles.btn : styles.btnDisabled} onClick={sendOnce} disabled={!canSend}>
          Generate (one-shot)
        </button>
      </div>

      {error ? <div style={styles.error}>⚠ {error}</div> : null}

      <div ref={scrollRef} style={styles.chatBox}>
        {messages.map((m, i) => (
          <div key={i} style={m.role === "assistant" ? styles.assistant : m.role === "user" ? styles.user : styles.system}>
            <div style={styles.role}>{m.role}</div>
            <pre style={styles.msg}>{m.text}</pre>
          </div>
        ))}
        {busy ? <div style={styles.busy}>… generating …</div> : null}
      </div>

      <div style={styles.footer}>
        <small>
          Endpoint: <code>{apiUrl}</code> • Mode: <code>{mode}</code> • Max tokens:{" "}
          <code>{maxNewTokens}</code>
        </small>
      </div>
    </div>
  );
}

const styles = {
  shell: { maxWidth: 820, margin: "16px auto", padding: 16, border: "1px solid #e5e7eb", borderRadius: 8, fontFamily: "Inter, system-ui, sans-serif" },
  header: { margin: "4px 0 16px 0" },
  formRow: { marginBottom: 8 },
  label: { display: "block", fontSize: 13, color: "#374151", marginBottom: 4 },
  textarea: { width: "100%", minHeight: 80, padding: 8, fontFamily: "inherit", borderRadius: 6, border: "1px solid #d1d5db", outline: "none" },
  btnRow: { display: "flex", gap: 8, margin: "8px 0 12px 0" },
  btn: { padding: "8px 12px", background: "#111827", color: "white", borderRadius: 6, border: "none", cursor: "pointer" },
  btnDisabled: { padding: "8px 12px", background: "#9ca3af", color: "white", borderRadius: 6, border: "none", cursor: "not-allowed" },
  error: { color: "#b91c1c", fontSize: 13, margin: "6px 0 10px 0" },
  chatBox: { border: "1px solid #e5e7eb", borderRadius: 6, padding: 8, minHeight: 200, maxHeight: 420, overflowY: "auto", background: "#fafafa" },
  system: { background: "#eef2ff", padding: 8, borderRadius: 6, marginBottom: 8 },
  user: { background: "#eff6ff", padding: 8, borderRadius: 6, marginBottom: 8 },
  assistant: { background: "#ecfdf5", padding: 8, borderRadius: 6, marginBottom: 8 },
  role: { fontSize: 11, fontWeight: 600, color: "#374151", marginBottom: 6, textTransform: "uppercase" },
  msg: { margin: 0, whiteSpace: "pre-wrap" },
  busy: { marginTop: 8, color: "#6b7280", fontStyle: "italic" },
  footer: { marginTop: 12, textAlign: "right", color: "#6b7280" },
};