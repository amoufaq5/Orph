import { useState } from "react";

export default function ImageVQA() {
  const [file, setFile] = useState(null);
  const [resp, setResp] = useState(null);
  const [alpha, setAlpha] = useState(0.5);

  async function submit() {
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch("/api/vqa?role=clinician", { method: "POST", body: fd });
    const j = await r.json();
    setResp(j);
  }

  return (
    <div style={{maxWidth:900, margin:"24px auto", fontFamily:"system-ui"}}>
      <h2>Imaging VQA (research)</h2>
      <input type="file" accept="image/*" onChange={e=>setFile(e.target.files?.[0] || null)} />
      <button onClick={submit} style={{marginLeft:8}}>Analyze</button>
      {resp && (
        <div style={{marginTop:16}}>
          <div style={{display:"flex", gap:16}}>
            <div style={{position:"relative", width:512}}>
              {/* Base image */}
              <img
                alt="input"
                src={URL.createObjectURL(file)}
                style={{width:"100%", display:"block", borderRadius:12}}
              />
              {/* Heatmap overlay */}
              <img
                alt="heatmap"
                src={`data:image/png;base64,${resp.heatmap_png_b64}`}
                style={{
                  position:"absolute", inset:0, width:"100%",
                  mixBlendMode:"multiply", opacity: alpha, borderRadius:12
                }}
              />
            </div>
            <div style={{flex:1}}>
              <h3>Result</h3>
              <p><b>Finding</b>: {resp.finding}</p>
              <p><b>Probability</b>: {(resp.probability*100).toFixed(1)}%</p>
              <p style={{opacity:0.8}}>{resp.disclaimer}</p>
              <label>Overlay opacity: {alpha.toFixed(2)}</label>
              <input type="range" min="0" max="1" step="0.01" value={alpha} onChange={e=>setAlpha(parseFloat(e.target.value))} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
