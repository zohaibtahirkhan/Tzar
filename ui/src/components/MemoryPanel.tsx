/**
 * src/components/MemoryPanel.tsx
 * Memory cards with scoring bars, add-memory form, search, prune.
 */
import React, { useEffect, useState } from "react";
import { Search, Trash2, Star, RefreshCw, Brain, Plus, X } from "lucide-react";
import { useStore } from "../stores/useStore";
import { getMemories, deleteMemory, getMemoryScores, pruneMemories, saveMemory } from "../api";
import type { Memory } from "../api";

function ScoreBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="score-bar-row">
      <span className="score-label">{label}</span>
      <div className="score-track">
        <div className="score-fill" style={{ width: `${Math.round((value ?? 0) * 100)}%`, background: color }} />
      </div>
      <span className="score-val">{((value ?? 0) * 100).toFixed(0)}</span>
    </div>
  );
}

function MemoryCard({ memory, onDelete }: { memory: Memory; onDelete: () => void }) {
  const [expanded, setExpanded] = useState(false);
  const imp = memory.importance ?? 0.5;
  const rec = memory.recency ?? 0.5;
  const com = memory.composite ?? 0.5;
  const impColor = imp >= 0.8 ? "var(--accent-purple)" : imp >= 0.5 ? "var(--accent-cyan)" : "var(--muted)";

  return (
    <div className={`memory-card ${expanded ? "memory-card-open" : ""}`} onClick={() => setExpanded(e => !e)}>
      <div className="memory-card-top">
        <div className="memory-meta">
          <span className="memory-category">{memory.category}</span>
          {imp >= 0.8 && <Star size={10} fill="var(--accent-amber)" color="var(--accent-amber)" />}
        </div>
        <button className="btn-icon delete-btn" onClick={e => { e.stopPropagation(); onDelete(); }}>
          <Trash2 size={12} />
        </button>
      </div>
      <p className="memory-content">{memory.content}</p>
      {expanded && (
        <div className="memory-scores" onClick={e => e.stopPropagation()}>
          <ScoreBar label="Importance" value={imp} color={impColor} />
          <ScoreBar label="Recency"    value={rec} color="var(--accent-cyan)" />
          <ScoreBar label="Composite"  value={com} color="var(--accent-green)" />
          {memory.frequency !== undefined && (
            <div className="score-bar-row">
              <span className="score-label">Recalled</span>
              <span className="score-val" style={{ color: "var(--accent-amber)" }}>{memory.frequency}×</span>
            </div>
          )}
          {memory.created_at && (
            <div className="score-bar-row">
              <span className="score-label">Created</span>
              <span className="score-val" style={{ width: "auto" }}>{memory.created_at.slice(0, 10)}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const CATEGORIES = ["all", "preference", "note", "fact", "project", "auto-learned"];

export function MemoryPanel() {
  const { memories, memoriesLoading, setMemories, setMemoriesLoading, removeMemory } = useStore();
  const [query, setQuery]       = useState("");
  const [category, setCategory] = useState("all");
  const [statsText, setStatsText] = useState("");
  const [showAdd, setShowAdd]   = useState(false);
  const [newCat, setNewCat]     = useState("note");
  const [newContent, setNewContent] = useState("");

  const load = async (q = query, cat = category) => {
    setMemoriesLoading(true);
    try {
      const res = await getMemories(q, cat === "all" ? "" : cat);
      setMemories(res.memories ?? []);
    } catch { /* ignore */ } finally { setMemoriesLoading(false); }
  };

  const loadStats = async () => {
    try { const r = await getMemoryScores(); setStatsText(r.result ?? ""); } catch { /* ignore */ }
  };

  useEffect(() => { load(); loadStats(); }, []);

  const handleDelete = async (id: number) => {
    await deleteMemory(id);
    removeMemory(id);
  };

  const handlePrune = async () => {
    await pruneMemories();
    load(); loadStats();
  };

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newContent.trim()) return;
    await saveMemory(newCat, newContent.trim());
    setNewContent(""); setShowAdd(false);
    load(); loadStats();
  };

  return (
    <div className="panel-wrap">
      <div className="panel-header">
        <Brain size={17} />
        <h2>Memory</h2>
        <div className="panel-header-actions">
          <button className="btn-small" onClick={() => setShowAdd(s => !s)}>
            {showAdd ? <><X size={11} /> Cancel</> : <><Plus size={11} /> Add</>}
          </button>
          <button className="btn-small" onClick={handlePrune}>Prune</button>
          <button className="btn-icon" onClick={() => { load(); loadStats(); }}><RefreshCw size={13} /></button>
        </div>
      </div>

      {/* Add memory form */}
      {showAdd && (
        <form className="add-memory-form" onSubmit={handleAdd}>
          <select className="cat-select" value={newCat} onChange={e => setNewCat(e.target.value)}>
            {CATEGORIES.filter(c => c !== "all").map(c => <option key={c}>{c}</option>)}
          </select>
          <textarea
            className="add-memory-input"
            value={newContent}
            onChange={e => setNewContent(e.target.value)}
            placeholder="Memory content…"
            rows={2}
          />
          <button type="submit" className="btn-small" disabled={!newContent.trim()}>Save</button>
        </form>
      )}

      {/* Stats */}
      {statsText && (
        <div className="stats-box">
          <pre className="stats-pre">{statsText}</pre>
        </div>
      )}

      {/* Search */}
      <form className="search-bar" onSubmit={e => { e.preventDefault(); load(); }}>
        <Search size={13} className="search-icon" />
        <input
          className="search-input"
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Search memories…"
        />
        <button type="submit" className="btn-small">Go</button>
      </form>

      {/* Category tabs */}
      <div className="cat-tabs">
        {CATEGORIES.map(c => (
          <button
            key={c}
            className={`cat-tab ${category === c ? "cat-tab-active" : ""}`}
            onClick={() => { setCategory(c); load(query, c); }}
          >{c}</button>
        ))}
      </div>

      {/* Cards */}
      <div className="memory-list">
        {memoriesLoading ? (
          <div className="panel-loading">Loading…</div>
        ) : memories.length === 0 ? (
          <div className="panel-empty">No memories found.</div>
        ) : (
          memories.map(m => (
            <MemoryCard key={m.id} memory={m} onDelete={() => handleDelete(m.id)} />
          ))
        )}
      </div>
    </div>
  );
}
