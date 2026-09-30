/**
 * src/components/ChatPanel.tsx
 * Streaming SSE chat, WebSocket voice, expandable tool results, stop button.
 */
import React, { useEffect, useRef, useState, useCallback } from "react";
import { Mic, MicOff, Send, Trash2, Square, ChevronDown, ChevronRight, Zap, FileText, StickyNote } from "lucide-react";
import { useStore, nextId, type Message } from "../stores/useStore";
import { streamChat, type Source } from "../api";

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

// ─── Sources (where a retrieval-grounded answer came from) ───────────────────

function SourcesRow({ sources }: { sources: Source[] }) {
  return (
    <div className="sources-row">
      <span className="sources-label">Sources</span>
      {sources.map((s, i) => (
        <span key={i} className="stat-chip source-chip" title={s.location}>
          {s.kind === "note" ? <StickyNote size={10} /> : <FileText size={10} />}
          {s.title}
        </span>
      ))}
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
        {msg.sources && msg.sources.length > 0 && <SourcesRow sources={msg.sources} />}
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

function useVoice(
  onTranscript: (t: string) => void, 
  onResponse: (r: string) => void,
  onStatus: (s: string) => void
) {
  const [listening, setListening] = useState(false);
  const wsRef     = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const inputCtxRef  = useRef<AudioContext | null>(null);   // input audio context
  const outputCtxRef = useRef<AudioContext | null>(null);   // output audio context for TTS
  const audioQueueRef = useRef<Float32Array[]>([]);         // TTS audio queue
  const playingRef = useRef(false);                         // TTS playback state

  // TTS playback function
  const playAudioQueue = useCallback(async () => {
    if (playingRef.current || audioQueueRef.current.length === 0) return;
    
    playingRef.current = true;
    
    try {
      // Create output AudioContext if not exists (without forcing sample rate)
      if (!outputCtxRef.current || outputCtxRef.current.state === 'closed') {
        outputCtxRef.current = new AudioContext();
      }
      
      const ctx = outputCtxRef.current;
      
      // Play all queued audio chunks
      while (audioQueueRef.current.length > 0) {
        const chunk = audioQueueRef.current.shift();
        if (!chunk) continue;
        
        // Create a new Float32Array backed by a regular ArrayBuffer
        // This is required because copyToChannel doesn't accept SharedArrayBuffer
        const audioData = new Float32Array(chunk.length);
        audioData.set(chunk);
        
        // Create buffer with TTS sample rate (24kHz)
        const buffer = ctx.createBuffer(1, audioData.length, 24000);
        buffer.copyToChannel(audioData, 0);
        
        const source = ctx.createBufferSource();
        source.buffer = buffer;
        source.connect(ctx.destination);
        
        // Wait for this chunk to finish
        await new Promise<void>((resolve) => {
          source.onended = () => resolve();
          source.start();
        });
      }
    } catch (err) {
      console.error("Audio playback error:", err);
    } finally {
      playingRef.current = false;
    }
  }, []);

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      // Create AudioContext without forcing sample rate (let browser/system decide)
      const ctx = new AudioContext();
      inputCtxRef.current = ctx;
      const src  = ctx.createMediaStreamSource(stream);
      
      // Use deprecated but more compatible ScriptProcessor
      // Modern alternative would be AudioWorklet but ScriptProcessor is more reliable
      const proc = ctx.createScriptProcessor(512, 1, 1);
      
      const ws = new WebSocket("ws://127.0.0.1:8000/ws/audio");
      wsRef.current = ws;
      ws.binaryType = "arraybuffer";
      
      ws.onopen = () => {
        console.log("WebSocket connected");
        proc.onaudioprocess = (e) => {
          if (ws.readyState === WebSocket.OPEN) {
            const inputData = e.inputBuffer.getChannelData(0);
            
            // Resample to 16kHz if needed
            if (ctx.sampleRate !== 16000) {
              // Simple downsampling (for production, use a proper resampler)
              const ratio = ctx.sampleRate / 16000;
              const targetLength = Math.floor(inputData.length / ratio);
              const resampled = new Float32Array(targetLength);
              
              for (let i = 0; i < targetLength; i++) {
                const srcIndex = Math.floor(i * ratio);
                resampled[i] = inputData[srcIndex];
              }
              
              ws.send(resampled.buffer);
            } else {
              ws.send(inputData.buffer);
            }
          }
        };
        src.connect(proc);
        proc.connect(ctx.destination);
        setListening(true);
      };
      
      ws.onmessage = (e) => {
        if (typeof e.data === "string") {
          // JSON messages
          try {
            const msg = JSON.parse(e.data);
            console.log("WebSocket message:", msg);
            
            if (msg.type === "transcript" && msg.data) {
              onTranscript(msg.data);
            } else if (msg.type === "response" && msg.data) {
              onResponse(msg.data);
            } else if (msg.type === "status" && msg.data) {
              onStatus(msg.data);
            } else if (msg.type === "error") {
              console.error("WebSocket error:", msg.data);
              onStatus(`Error: ${msg.data}`);
            }
          } catch (err) {
            console.error("JSON parse error:", err);
          }
        } else if (e.data instanceof ArrayBuffer) {
          // Binary audio data from TTS
          try {
            const audioData = new Float32Array(e.data);
            audioQueueRef.current.push(audioData);
            playAudioQueue();
          } catch (err) {
            console.error("Audio playback error:", err);
          }
        }
      };
      
      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        onStatus("Connection error");
      };
      
      ws.onclose = () => {
        console.log("WebSocket closed");
        setListening(false);
      };
      
    } catch (err) {
      console.error("Mic error:", err);
      const errorMsg = err instanceof Error ? err.message : String(err);
      alert(`Microphone access error: ${errorMsg}\n\nPlease grant microphone permissions.`);
    }
  }, [onTranscript, onResponse, onStatus, playAudioQueue]);

  const stop = useCallback(() => {
    wsRef.current?.close();
    streamRef.current?.getTracks().forEach(t => t.stop());
    inputCtxRef.current?.close();
    outputCtxRef.current?.close();
    inputCtxRef.current = null;
    outputCtxRef.current = null;
    audioQueueRef.current = [];
    playingRef.current = false;
    setListening(false);
  }, []);

  return { listening, start, stop };
}

// ─── Chat Panel ───────────────────────────────────────────────────────────────

export function ChatPanel() {
  const {
    messages, isStreaming, inputText, setInputText,
    addMessage, appendToken, appendToolResults, setSources, finaliseStream, clearChat,
  } = useStore();

  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef  = useRef<AbortController | null>(null);
  const [voiceStatus, setVoiceStatus] = useState<string>("");

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Voice callbacks
  const handleTranscript = useCallback((transcript: string) => {
    console.log("Transcript received:", transcript);
    // Add user message
    addMessage({ 
      id: nextId(), 
      role: "user", 
      content: transcript, 
      ts: Date.now() 
    });
  }, [addMessage]);

  const handleResponse = useCallback((response: string) => {
    console.log("Response received:", response);
    // Add assistant message
    addMessage({ 
      id: nextId(), 
      role: "assistant", 
      content: response, 
      ts: Date.now() 
    });
    setVoiceStatus("");  // Clear status
  }, [addMessage]);

  const handleStatus = useCallback((status: string) => {
    console.log("Status:", status);
    setVoiceStatus(status);
  }, []);

  const { listening, start, stop } = useVoice(
    handleTranscript,
    handleResponse,
    handleStatus
  );

  const submit = useCallback((text: string) => {
    if (!text.trim() || isStreaming) return;
    setInputText("");

    addMessage({ id: nextId(), role: "user", content: text.trim(), ts: Date.now() });

    const aId = nextId();
    addMessage({ id: aId, role: "assistant", content: "", streaming: true, toolResults: [], ts: Date.now() });

    abortRef.current = streamChat(
      text.trim(),
      (token)   => appendToken(aId, token),
      (results) => appendToolResults(aId, results),
      ()        => finaliseStream(aId),
      (err)     => { appendToken(aId, `\n\n[Error: ${err}]`); finaliseStream(aId); },
      (sources) => setSources(aId, sources)
    );
  }, [isStreaming, addMessage, appendToken, appendToolResults, setSources, finaliseStream, setInputText]);

  const stopStream = () => {
    abortRef.current?.abort();
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
          {voiceStatus ? `${voiceStatus}…` : "Listening… press mic to stop"}
        </div>
      )}
    </div>
  );
}
