/**
 * src/components/ChatPanel.tsx
 * Streaming SSE chat, WebSocket voice, expandable tool results, stop button.
 */
import React, { useEffect, useRef, useState, useCallback } from "react";
import { Mic, MicOff, Send, Trash2, Square, ChevronDown, ChevronRight, Zap } from "lucide-react";
import { useStore, nextId, type Message } from "../stores/useStore";
import { streamChat } from "../api";

// ─── Tool result expander ─────────────────────────────────────────────────────

function ToolResultRow({ tool, result, status }: { tool: string; result: string; status: string }) {
  const [open, setOpen] = useState(false);
  const ok = status === "ok";
  return (
    <div className={`tool-result-row ${ok ? "tool-ok" : "tool-err"}`}>
      <button className="tool-result-header" onClick={() => setOpen(o => !o)}>
        <Zap size={10} />
        <span className="tool-name">{tool}</span>
        <span className={`tool-status ${ok ? "ts-ok" : "ts-err"}`}>{ok ? "ok" : "error"}</span>
        {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
      </button>
      {open && result && (
        <pre className="tool-result-body">{result.slice(0, 600)}{result.length > 600 ? "\n…" : ""}</pre>
      )}
    </div>
  );
}

// ─── Message bubble ───────────────────────────────────────────────────────────

function Bubble({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";

  if (msg.role === "system") {
    return <div className="sys-msg">{msg.content}</div>;
  }

  return (
    <div className={`bubble-wrap ${isUser ? "bubble-user" : "bubble-assistant"}`}>
      <div className={`bubble ${isUser ? "bubble-u" : "bubble-a"}`}>
        <p className="bubble-text">{msg.content || "\u00a0"}</p>
        {msg.streaming && <span className="cursor-blink" />}
        {msg.toolResults && msg.toolResults.length > 0 && (
          <div className="tool-results-list">
            {msg.toolResults.map((r, i) => (
              <ToolResultRow key={i} tool={r.tool} result={r.result} status={r.status} />
            ))}
          </div>
        )}
      </div>
      <time className="bubble-time">
        {new Date(msg.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
      </time>
    </div>
  );
}

// ─── Voice hook ───────────────────────────────────────────────────────────────

function useVoice(onTranscript: (t: string) => void) {
  const [listening, setListening] = useState(false);
  const wsRef    = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const ctx = new AudioContext({ sampleRate: 16000 });
      const src = ctx.createMediaStreamSource(stream);
      const proc = ctx.createScriptProcessor(512, 1, 1);
      const ws = new WebSocket("ws://127.0.0.1:8000/ws/audio");
      wsRef.current = ws;
      ws.binaryType = "arraybuffer";
      ws.onopen = () => {
        proc.onaudioprocess = (e) => {
          if (ws.readyState === WebSocket.OPEN)
            ws.send(e.inputBuffer.getChannelData(0).buffer);
        };
        src.connect(proc);
        proc.connect(ctx.destination);
        setListening(true);
      };
      ws.onmessage = (e) => {
        if (typeof e.data === "string") {
          try {
            const msg = JSON.parse(e.data);
            if (msg.type === "transcript" && msg.data) onTranscript(msg.data);
          } catch { /* ignore */ }
        }
      };
      ws.onclose = () => setListening(false);
    } catch (err) { console.error("Mic error:", err); }
  }, [onTranscript]);

  const stop = useCallback(() => {
    wsRef.current?.close();
    streamRef.current?.getTracks().forEach(t => t.stop());
    setListening(false);
  }, []);

  return { listening, start, stop };
}

// ─── Chat Panel ───────────────────────────────────────────────────────────────

export function ChatPanel() {
  const {
    messages, isStreaming, inputText, setInputText,
    addMessage, appendToken, finaliseStream, clearChat,
  } = useStore();

  const bottomRef  = useRef<HTMLDivElement>(null);
  const abortRef   = useRef<AbortController | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const { listening, start, stop } = useVoice(useCallback((t) => setInputText(t), [setInputText]));

  const submit = useCallback((text: string) => {
    if (!text.trim() || isStreaming) return;
    setInputText("");

    addMessage({ id: nextId(), role: "user", content: text.trim(), ts: Date.now() });

    const aId = nextId();
    addMessage({ id: aId, role: "assistant", content: "", streaming: true, ts: Date.now() });

    abortRef.current = streamChat(
      text.trim(),
      (token) => appendToken(aId, token),
      () => finaliseStream(aId),
      (err) => { appendToken(aId, `\n\n[Error: ${err}]`); finaliseStream(aId); }
    );
  }, [isStreaming, addMessage, appendToken, finaliseStream, setInputText]);

  const stopStream = () => {
    abortRef.current?.abort();
    // finalise the last streaming message
    const last = [...messages].reverse().find(m => m.streaming);
    if (last) finaliseStream(last.id);
  };

  const handleKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(inputText); }
  };

  return (
    <div className="chat-panel">
      <div className="chat-messages">
        {messages.map(m => <Bubble key={m.id} msg={m} />)}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-bar">
        <button
          className={`btn-icon mic-btn ${listening ? "mic-active" : ""}`}
          onClick={listening ? stop : start}
          title={listening ? "Stop recording" : "Voice input"}
        >
          {listening ? <MicOff size={17} /> : <Mic size={17} />}
        </button>

        <textarea
          className="chat-textarea"
          value={inputText}
          onChange={e => setInputText(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Message Tzar… (Enter to send, Shift+Enter for newline)"
          rows={1}
          disabled={isStreaming}
        />

        {isStreaming ? (
          <button className="btn-icon stop-btn" onClick={stopStream} title="Stop">
            <Square size={16} />
          </button>
        ) : (
          <button
            className="btn-icon send-btn"
            onClick={() => submit(inputText)}
            disabled={!inputText.trim()}
            title="Send"
          >
            <Send size={17} />
          </button>
        )}

        <button className="btn-icon clear-btn" onClick={clearChat} title="Clear chat">
          <Trash2 size={15} />
        </button>
      </div>

      {listening && (
        <div className="listening-bar">
          <span className="listening-dot" />
          Listening…  press mic to stop
        </div>
      )}
    </div>
  );
}
