// App.jsx (Single File React App)
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = 'http://localhost:5000';

function App() {
  const [view, setView] = useState('login');
  const [token, setToken] = useState(localStorage.getItem('token') || '');
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');
  const [history, setHistory] = useState([]);

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
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>Orph AI Chat</h1>

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
          ></textarea>
          <button onClick={sendMessage}>Send</button>
          <button onClick={() => { localStorage.removeItem('token'); setView('login'); }}>Logout</button>
          <pre>{response}</pre>
          <h3>Chat History</h3>
          <ul>
            {history.map((h, i) => (
              <li key={i}><strong>You:</strong> {h.input}<br /><strong>AI:</strong> {JSON.stringify(h.output)}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
