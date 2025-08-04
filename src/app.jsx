// App.jsx (Extended MVP with upload, charts, and routing)
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import 'chart.js/auto';

const API_BASE = 'http://localhost:5000';

function App() {
  const [view, setView] = useState('login');
  const [token, setToken] = useState(localStorage.getItem('token') || '');
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');
  const [history, setHistory] = useState([]);
  const [symptomFreq, setSymptomFreq] = useState({});
  const [file, setFile] = useState(null);
  const [meta, setMeta] = useState({ age: '', duration_days: '', danger_symptoms: '' });

  const handleLogin = async (e) => {
    e.preventDefault();
    const { username, password } = e.target.elements;
    const res = await axios.post(`${API_BASE}/login`, {
      username: username.value,
      password: password.value
    });
    if (res.data.token) {
      localStorage.setItem('token', res.data.token);
      setToken(res.data.token);
      setView('chat');
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    const { username, password } = e.target.elements;
    await axios.post(`${API_BASE}/register`, {
      username: username.value,
      password: password.value
    });
    alert('Registration successful. Please log in.');
    setView('login');
  };

  const sendMessage = async () => {
    const res = await axios.post(`${API_BASE}/chat`, { message }, {
      headers: { Authorization: token }
    });
    setResponse(JSON.stringify(res.data.response, null, 2));
    setHistory([...history, { input: message, output: res.data.response }]);
    setMessage('');
    updateChart(res.data.response);
  };

  const uploadImage = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    formData.append('text', message);
    formData.append('meta', JSON.stringify({
      age: parseInt(meta.age),
      duration_days: parseInt(meta.duration_days),
      danger_symptoms: meta.danger_symptoms.split(',').map(s => s.trim())
    }));

    const res = await axios.post(`${API_BASE}/upload`, formData, {
      headers: {
        Authorization: token,
        'Content-Type': 'multipart/form-data'
      }
    });
    alert(`Upload complete. ${res.data.message}`);
    setResponse(JSON.stringify(res.data.predictions, null, 2));
    updateChart(res.data.predictions);
  };

  const updateChart = (output) => {
    if (!Array.isArray(output)) return;
    const symptoms = output.map(r => r.disease || r.name || `Disease ${r[0]}`);
    const updated = { ...symptomFreq };
    symptoms.forEach(sym => {
      updated[sym] = (updated[sym] || 0) + 1;
    });
    setSymptomFreq(updated);
  };

  const chartData = {
    labels: Object.keys(symptomFreq),
    datasets: [{
      label: 'Prediction Frequency',
      data: Object.values(symptomFreq),
      fill: false,
      borderColor: 'blue'
    }]
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>Orph AI Dashboard</h1>

      {view === 'login' && (
        <form onSubmit={handleLogin}>
          <input name="username" placeholder="Username" /><br />
          <input name="password" placeholder="Password" type="password" /><br />
          <button type="submit">Login</button>
          <p>or <button type="button" onClick={() => setView('register')}>Register</button></p>
        </form>
      )}

      {view === 'register' && (
        <form onSubmit={handleRegister}>
          <input name="username" placeholder="Username" /><br />
          <input name="password" placeholder="Password" type="password" /><br />
          <button type="submit">Register</button>
          <p>or <button type="button" onClick={() => setView('login')}>Back to Login</button></p>
        </form>
      )}

      {view === 'chat' && (
        <div>
          <textarea
            rows="3"
            style={{ width: '100%' }}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Describe your symptoms..."
          ></textarea><br />

          <input type="number" placeholder="Age" value={meta.age} onChange={e => setMeta({ ...meta, age: e.target.value })} />
          <input type="number" placeholder="Duration in days" value={meta.duration_days} onChange={e => setMeta({ ...meta, duration_days: e.target.value })} />
          <input type="text" placeholder="Danger symptoms (comma separated)" value={meta.danger_symptoms} onChange={e => setMeta({ ...meta, danger_symptoms: e.target.value })} /><br />

          <button onClick={sendMessage}>Send</button>
          <input type="file" onChange={(e) => setFile(e.target.files[0])} /><button onClick={uploadImage}>Upload CT/X-ray</button>
          <button onClick={() => { localStorage.removeItem('token'); setToken(''); setView('login'); }}>Logout</button>
          <pre>{response}</pre>

          <h3>Chat History</h3>
          <ul>
            {history.map((h, i) => (
              <li key={i}><strong>You:</strong> {h.input}<br /><strong>AI:</strong> {JSON.stringify(h.output)}</li>
            ))}
          </ul>

          <h3>Prediction Trends</h3>
          <Line data={chartData} />
        </div>
      )}
    </div>
  );
}

export default App;
