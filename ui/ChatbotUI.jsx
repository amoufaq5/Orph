// ChatbotUI.jsx
import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

export default function ChatbotUI() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

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

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center p-4">
      <h1 className="text-2xl font-semibold mb-4">🩺 Orph Medical Chatbot</h1>
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
      </div>
    </div>
  );
}
