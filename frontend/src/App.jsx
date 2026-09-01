import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { checkHealth, askStreaming, uploadDocument } from "./api";
import "./index.css";

export default function App() {
  const [health, setHealth] = useState(null);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  // const [retrieveN, setRetrieveN] = useState("");
  // const [rerankK, setRerankK] = useState("");
  const [uploadStatus, setUploadStatus] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const bottomRef = useRef(null);
  const fileInputRef = useRef(null);

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
        // retrieveN: retrieveN ? Number(retrieveN) : undefined,
        // rerankK: rerankK ? Number(rerankK) : undefined,
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
              text: `Trail went cold: ${event.message}`,
            };
            return next;
          });
        }
      },
    );

    setIsStreaming(false);
  }

  async function handleFileSelect(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadStatus({
      type: "pending",
      text: `Filing "${file.name}" into evidence…`,
    });

    try {
      const result = await uploadDocument(file);
      setUploadStatus({
        type: "ok",
        text: `Case file logged: ${result.chunks_stored} clues extracted from ${result.pages} pages.`,
      });
      checkHealth().then(setHealth);
    } catch (err) {
      setUploadStatus({ type: "error", text: `Filing failed: ${err.message}` });
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  return (
    <div className="app">
      <header>
        <div className="brand">
          <span className="glass">🔍</span>
          <h1>Evidence Bureau</h1>
        </div>
        <span className={`status ${health?.status === "ok" ? "ok" : "down"}`}>
          {health?.status === "ok"
            ? `case open — ${health.model}`
            : "case closed"}
        </span>
      </header>

      <div className="chat">
        {messages.length === 0 && (
          <div className="empty-state">
            <p>
              No leads yet. File a document below, then ask your first question.
            </p>
          </div>
        )}
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
              "investigating…"
            ) : (
              ""
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="dossier">
        <input
          ref={fileInputRef}
          type="file"
          accept="application/pdf"
          onChange={handleFileSelect}
          disabled={isUploading}
          id="file-upload"
          hidden
        />
        <label
          htmlFor="file-upload"
          className={`upload-btn ${isUploading ? "disabled" : ""}`}
        >
          📁 {isUploading ? "Filing…" : "Add to case file"}
        </label>
        {uploadStatus && (
          <span className={`upload-status ${uploadStatus.type}`}>
            {uploadStatus.text}
          </span>
        )}
      </div>

      <form className="composer" onSubmit={handleAsk}>
        {/* <div className="params">
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
        </div> */}
        <div className="input-row">
          <input
            type="text"
            placeholder="Interrogate the evidence…"
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
