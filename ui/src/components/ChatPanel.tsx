/**
 * src/components/ChatPanel.tsx
 *
 * Main chat interface with SSE streaming, mic input, and tool result display.
 */

import React, { useEffect, useRef, useState, useCallback } from "react";
import { Mic, MicOff, Send, Trash2, Zap } from "lucide-react";
import { useStore, nextId, type Message } from "../stores/useStore";
import { streamChat } from "../api";

// ─── Message bubble ───────────────────────────────────────────────────────────

function Bubble({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";
  const isSystem = msg.role === "system";

  if (isSystem) {
    return (
      <div className="sys-msg">
        <span>{msg.content}</span>
      </div>
    );
  }

  return (
    <div className={`bubble-wrap ${isUser ? "bubble-user" : "bubble-assistant"}`}>
      <div className={`bubble ${isUser ? "bubble-u" : "bubble-a"}`}>
        {msg.streaming && (
          <span className="cursor-blink" aria-hidden="true" />
        )}
        <p className="bubble-text">{msg.content || "\u00a0"}</p>
        {msg.toolResults && msg.toolResults.length > 0 && (
          <div className="tool-results">
            {msg.toolResults.map((r, i) => (
              <div key={i} className="tool-chip">
                <Zap size={10} />
                <span className="tool-name">{r.tool}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      <time className="bubble-time">
        {new Date(msg.ts).toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        })}
      </time>
    </div>
  );
}

// ─── WebSocket audio hook ─────────────────────────────────────────────────────

function useVoiceInput(onTranscript: (text: string) => void) {
  const [listening, setListening] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRef.current = stream;

      const AudioContextClass =
        window.AudioContext ||
        (
          window as Window & {
            webkitAudioContext?: typeof window.AudioContext;
          }
        ).webkitAudioContext;

      if (!AudioContextClass) {
        throw new Error("AudioContext not supported");
      }

      const ctx = new AudioContextClass({
        sampleRate: 16000,
      });
      const source = ctx.createMediaStreamSource(stream);
      const processor = ctx.createScriptProcessor(512, 1, 1);
      processorRef.current = processor;

      const ws = new WebSocket("ws://127.0.0.1:8000/ws/audio");
      wsRef.current = ws;
      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        processor.onaudioprocess = (e: AudioProcessingEvent) => {
          if (ws.readyState !== WebSocket.OPEN) return;
          const pcm = e.inputBuffer.getChannelData(0);
          ws.send(pcm.buffer);
        };
        source.connect(processor);
        processor.connect(ctx.destination);
        setListening(true);
      };

      ws.onmessage = (e) => {
        if (typeof e.data === "string") {
          try {
            const msg = JSON.parse(e.data);
            if (msg.type === "transcript" && msg.data) {
              onTranscript(msg.data);
            }
            if (msg.type === "response" && msg.data) {
              // Response will be handled by the SSE stream in parallel
            }
          } catch {
            /* ignore */
          }
        }
      };

      ws.onclose = () => setListening(false);
    } catch (err) {
      console.error("Microphone error:", err);
    }
  }, [onTranscript]);

  const stop = useCallback(() => {
    wsRef.current?.close();
    mediaRef.current?.getTracks().forEach((t) => t.stop());
    processorRef.current?.disconnect();
    wsRef.current = null;
    mediaRef.current = null;
    setListening(false);
  }, []);

  return { listening, start, stop };
}

// ─── Chat Panel ───────────────────────────────────────────────────────────────

export function ChatPanel() {
  const {
    messages,
    isStreaming,
    inputText,
    setInputText,
    addMessage,
    appendToken,
    finaliseStream,
    clearChat,
  } = useStore();

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleTranscript = useCallback(
    (text: string) => {
      setInputText(text);
    },
    [setInputText]
  );

  const { listening, start, stop } = useVoiceInput(handleTranscript);

  const submit = useCallback(
    (text: string) => {
      if (!text.trim() || isStreaming) return;
      setInputText("");

      // Add user message
      addMessage({
        id: nextId(),
        role: "user",
        content: text.trim(),
        ts: Date.now(),
      });

      // Add empty assistant message for streaming
      const assistantId = nextId();
      addMessage({
        id: assistantId,
        role: "assistant",
        content: "",
        streaming: true,
        ts: Date.now(),
      });

      // Start SSE stream
      abortRef.current = streamChat(
        text.trim(),
        (token) => appendToken(assistantId, token),
        () => finaliseStream(assistantId),
        (err) => {
          appendToken(assistantId, `\n\n[Error: ${err}]`);
          finaliseStream(assistantId);
        }
      );
    },
    [isStreaming, addMessage, appendToken, finaliseStream, setInputText]
  );

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit(inputText);
    }
  };

  return (
    <div className="chat-panel">
      {/* Messages */}
      <div className="chat-messages">
        {messages.map((m) => (
          <Bubble key={m.id} msg={m} />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="chat-input-bar">
        <button
          className={`btn-icon mic-btn ${listening ? "mic-active" : ""}`}
          onClick={listening ? stop : start}
          title={listening ? "Stop recording" : "Start voice input"}
        >
          {listening ? <MicOff size={18} /> : <Mic size={18} />}
        </button>

        <textarea
          ref={inputRef}
          className="chat-textarea"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Message Hafiz…"
          rows={1}
          disabled={isStreaming}
        />

        <button
          className="btn-icon send-btn"
          onClick={() => submit(inputText)}
          disabled={!inputText.trim() || isStreaming}
          title="Send (Enter)"
        >
          <Send size={18} />
        </button>

        <button
          className="btn-icon clear-btn"
          onClick={clearChat}
          title="Clear chat"
        >
          <Trash2 size={16} />
        </button>
      </div>

      {listening && (
        <div className="listening-indicator">
          <span className="listening-dot" />
          Listening…
        </div>
      )}
    </div>
  );
}
