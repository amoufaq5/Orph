// ChatbotUI.jsx with Chat History, PDF Export, and Profile Login
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
    if (!input.trim()) return;
    setMessages((prev) => [...prev, { sender: "user", text: input }]);
    setLoading(true);

    try {
      const res = await fetch("http://localhost:8080/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input }),
      });
      const data = await res.json();
      setMessages((prev) => [...prev, { sender: "bot", text: data.response }]);
    } catch (err) {
      setMessages((prev) => [...prev, { sender: "bot", text: "Error getting response." }]);
    }

    setInput("");
    setLoading(false);
  };

  const handleClear = () => {
    setMessages([]);
    localStorage.removeItem(CHAT_HISTORY_KEY);
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
          <Button variant="outline" onClick={handleLogout}>
            Logout
          </Button>
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
        </CardContent>
      </Card>

      <div className="flex w-full max-w-xl mt-4 gap-2">
        <Input
          placeholder="Enter symptoms..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
        />
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
  );
}
