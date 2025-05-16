// ChatbotUI.jsx with Grad-CAM display, file upload, history, login, and PDF export
import { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import jsPDF from "jspdf";

const CHAT_HISTORY_KEY = "orph_chat_history";
const USERNAME_KEY = "orph_username";

export default function ChatbotUI() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [username, setUsername] = useState("");
  const [loggedIn, setLoggedIn] = useState(false);
  const [file, setFile] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);

  useEffect(() => {
    const savedChat = localStorage.getItem(CHAT_HISTORY_KEY);
    if (savedChat) setMessages(JSON.parse(savedChat));
    const savedUser = localStorage.getItem(USERNAME_KEY);
    if (savedUser) {
      setUsername(savedUser);
      setLoggedIn(true);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(messages));
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() && !file) return;
    if (input.trim()) {
      setMessages((prev) => [...prev, { sender: "user", text: input }]);
    }
    if (file) {
      setMessages((prev) => [...prev, { sender: "user", text: `Uploaded file: ${file.name}` }]);
    }
    setLoading(true);

    try {
      const formData = new FormData();
      if (input.trim()) formData.append("input", input);
      if (file) formData.append("file", file);

      const res = await fetch("http://localhost:8080/chat", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setMessages((prev) => [...prev, { sender: "bot", text: data.response }]);
      if (data.heatmap) {
        setHeatmapUrl(`http://localhost:8080${data.heatmap}`);
      }
    } catch (err) {
      setMessages((prev) => [...prev, { sender: "bot", text: "Error getting response." }]);
    }

    setInput("");
    setFile(null);
    setLoading(false);
  };

  const handleClear = () => {
    setMessages([]);
    localStorage.removeItem(CHAT_HISTORY_KEY);
    setHeatmapUrl(null);
  };

  const handleExportPDF = () => {
    const doc = new jsPDF();
    doc.setFontSize(12);
    doc.text(`Chat History - ${username}`, 10, 10);
    let y = 20;
    messages.forEach((msg, idx) => {
      const prefix = msg.sender === "user" ? "You: " : "Orph: ";
      doc.text(`${prefix}${msg.text}`, 10, y);
      y += 10;
    });
    doc.save(`orph_chat_${username || "guest"}.pdf`);
  };

  const handleLogin = () => {
    if (username.trim()) {
      setLoggedIn(true);
      localStorage.setItem(USERNAME_KEY, username);
    }
  };

  const handleLogout = () => {
    setLoggedIn(false);
    setUsername("");
    localStorage.removeItem(USERNAME_KEY);
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center p-4">
      <h1 className="text-2xl font-semibold mb-4">🩺 Orph Medical Chatbot</h1>

      {!loggedIn ? (
        <div className="mb-4 flex gap-2">
          <Input
            placeholder="Enter your name to login..."
            value={username}
            onChange={(e) => setUsername(e.target.value)}
          />
          <Button onClick={handleLogin}>Login</Button>
        </div>
      ) : (
        <div className="mb-4 flex gap-2 items-center">
          <span className="font-medium">Welcome, {username}</span>
          <Button variant="outline" onClick={handleLogout}>Logout</Button>
        </div>
      )}

      <Card className="w-full max-w-xl shadow-md">
        <CardContent className="space-y-2 p-4 h-[500px] overflow-y-auto bg-white">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`p-2 rounded-xl ${
                msg.sender === "user" ? "bg-blue-100 text-right" : "bg-green-100 text-left"
              }`}
            >
              {msg.text}
            </div>
          ))}
          {heatmapUrl && (
            <div className="mt-4">
              <p className="text-sm text-gray-600">Grad-CAM Heatmap:</p>
              <img src={heatmapUrl} alt="Heatmap" className="mt-2 rounded-xl border" />
            </div>
          )}
        </CardContent>
      </Card>

      <div className="flex w-full max-w-xl mt-4 flex-col gap-2">
        <Input
          placeholder="Enter symptoms or notes..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
        />
        <input
          type="file"
          accept="image/*,.pdf,.txt"
          onChange={(e) => setFile(e.target.files[0])}
          className="block border rounded p-2 bg-white"
        />
        <div className="flex gap-2">
          <Button onClick={handleSend} disabled={loading}>
            {loading ? "Thinking..." : "Send"}
          </Button>
          <Button variant="outline" onClick={handleClear}>
            Clear
          </Button>
          <Button variant="secondary" onClick={handleExportPDF}>
            Export PDF
          </Button>
        </div>
      </div>
    </div>
  );
}
