const API_BASE = "http://localhost:8000";

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

// Streams the /ask endpoint. Calls onEvent(event) for each SSE message.
// event.event is one of: "start" | "token" | "done" | "error"
export async function askStreaming(
  question,
  { retrieveN, rerankK } = {},
  onEvent,
) {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      retrieve_n: retrieveN ?? null,
      rerank_k: rerankK ?? null,
      stream: true,
    }),
  });

  if (!res.ok || !res.body) {
    onEvent({ event: "error", message: `Request failed: ${res.status}` });
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n\n");
    buffer = lines.pop(); // keep the last (possibly incomplete) chunk

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const jsonStr = line.slice(6);
      try {
        onEvent(JSON.parse(jsonStr));
      } catch {
        // ignore malformed chunks
      }
    }
  }
}

export async function uploadDocument(file, onProgress) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/process_document`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Upload failed: ${res.status}`);
  }

  return res.json();
}