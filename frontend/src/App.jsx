import { useEffect, useRef, useState } from "react";
import { checkHealth, askStreaming } from "./api";
import "./index.css";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function App() {
  const [health, setHealth] = useState(null);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]); // { role: "user" | "assistant", text: string }
  const [isStreaming, setIsStreaming] = useState(false);
  const [retrieveN, setRetrieveN] = useState("");
  const [rerankK, setRerankK] = useState("");
  const bottomRef = useRef(null);

  useEffect(() => {
    checkHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: "error" }));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleAsk(e) {
    e.preventDefault();
    const q = question.trim();
    if (!q || isStreaming) return;

    setMessages((prev) => [
      ...prev,
      { role: "user", text: q },
      { role: "assistant", text: "" },
    ]);
    setQuestion("");
    setIsStreaming(true);

    await askStreaming(
      q,
      {
        retrieveN: retrieveN ? Number(retrieveN) : undefined,
        rerankK: rerankK ? Number(rerankK) : undefined,
      },
      (event) => {
        if (event.event === "token") {
          setMessages((prev) => {
            const next = [...prev];
            next[next.length - 1] = {
              role: "assistant",
              text: next[next.length - 1].text + event.content,
            };
            return next;
          });
        } else if (event.event === "error") {
          setMessages((prev) => {
            const next = [...prev];
            next[next.length - 1] = {
              role: "assistant",
              text: `Error: ${event.message}`,
            };
            return next;
          });
        }
      },
    );

    setIsStreaming(false);
  }

  return (
    <div className="app">
      <header>
        <h1>Evidence Bureau</h1>
        <span className={`status ${health?.status === "ok" ? "ok" : "down"}`}>
          {health?.status === "ok" ? `● online — ${health.model}` : "● offline"}
        </span>
      </header>

      <div className="chat">
        {messages.map((m, i) => (
          <div key={i} className={`bubble ${m.role}`}>
            {m.text ? (
              m.role === "assistant" ? (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {m.text}
                </ReactMarkdown>
              ) : (
                m.text
              )
            ) : isStreaming && i === messages.length - 1 ? (
              "…"
            ) : (
              ""
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <form className="composer" onSubmit={handleAsk}>
        <div className="params">
          <input
            type="number"
            placeholder="retrieve_n"
            value={retrieveN}
            onChange={(e) => setRetrieveN(e.target.value)}
          />
          <input
            type="number"
            placeholder="rerank_k"
            value={rerankK}
            onChange={(e) => setRerankK(e.target.value)}
          />
        </div>
        <div className="input-row">
          <input
            type="text"
            placeholder="Ask a question…"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={isStreaming}
          />
          <button type="submit" disabled={isStreaming}>
            {isStreaming ? "…" : "Ask"}
          </button>
        </div>
      </form>
    </div>
  );
}
