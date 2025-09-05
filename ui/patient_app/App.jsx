import { useState } from "react";

export default function App() {
  const [query, setQuery] = useState("");
  const [drugs, setDrugs] = useState("");
  const [resp, setResp] = useState(null);

  async function ask() {
    const body = {
      role: "patient",
      query,
      drugs: drugs ? drugs.split(",").map(s=>s.trim()) : []
    };
    const r = await fetch("/api/chat", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body) });
    setResp(await r.json());
  }

  return (
    <div style={{maxWidth:720, margin:"40px auto", fontFamily:"system-ui"}}>
      <h1>Orph Patient</h1>
      <p>Describe your symptoms. You can also enter medicines to check interactions.</p>
      <textarea rows={5} value={query} onChange={e=>setQuery(e.target.value)} style={{width:"100%"}} />
      <input placeholder="meds (comma-separated)" value={drugs} onChange={e=>setDrugs(e.target.value)} style={{width:"100%", marginTop:8}} />
      <button onClick={ask} style={{marginTop:12}}>Ask</button>
      {resp && (<div style={{marginTop:20, padding:12, border:"1px solid #3333"}}>
        <div><strong>Answer</strong></div>
        <pre style={{whiteSpace:"pre-wrap"}}>{resp.answer}</pre>
        <div style={{opacity:0.8, marginTop:8}}>{resp.disclaimer}</div>
      </div>)}
    </div>
  );
}
