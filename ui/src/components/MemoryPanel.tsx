/**
 * src/components/MemoryPanel.tsx
 *
 * Displays all long-term memories with scoring bars.
 * Supports search, category filter, and delete.
 */

import React, { useEffect, useState } from "react";
import { Search, Trash2, Star, Clock, RefreshCw, Brain } from "lucide-react";
import { useStore } from "../stores/useStore";
import { getMemories, deleteMemory, getMemoryScores, invokeTool } from "../api";
import type { Memory } from "../api";

// ─── Score bar ────────────────────────────────────────────────────────────────

function ScoreBar({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div className="score-bar-row">
      <span className="score-label">{label}</span>
      <div className="score-track">
        <div
          className="score-fill"
          style={{ width: `${Math.round(value * 100)}%`, background: color }}
        />
      </div>
      <span className="score-val">{(value * 100).toFixed(0)}</span>
    </div>
  );
}

// ─── Memory card ──────────────────────────────────────────────────────────────

function MemoryCard({ memory, onDelete }: { memory: Memory; onDelete: () => void }) {
  const [expanded, setExpanded] = useState(false);
  const importance = memory.importance ?? 0.5;
  const recency = memory.recency ?? 0.5;
  const composite = memory.composite ?? 0.5;

  const importanceColor =
    importance >= 0.8
      ? "var(--accent-purple)"
      : importance >= 0.5
      ? "var(--accent-cyan)"
      : "var(--muted)";

  return (
    <div
      className={`memory-card ${expanded ? "memory-card-expanded" : ""}`}
      onClick={() => setExpanded((e) => !e)}
    >
      <div className="memory-card-header">
        <div className="memory-meta">
          <span className="memory-category">{memory.category}</span>
          {importance >= 0.8 && (
            <Star size={11} className="importance-star" fill="currentColor" />
          )}
        </div>
        <button
          className="btn-icon delete-btn"
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          title="Delete memory"
        >
          <Trash2 size={13} />
        </button>
      </div>

      <p className="memory-content">{memory.content}</p>

      {expanded && (
        <div className="memory-scores" onClick={(e) => e.stopPropagation()}>
          <ScoreBar label="Importance" value={importance} color={importanceColor} />
          <ScoreBar label="Recency" value={recency} color="var(--accent-cyan)" />
          <ScoreBar
            label="Composite"
            value={composite}
            color="var(--accent-green)"
          />
          {memory.frequency !== undefined && (
            <div className="score-bar-row">
              <span className="score-label">Recalled</span>
              <span className="score-val freq-val">{memory.frequency}×</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Memory Panel ─────────────────────────────────────────────────────────────

const CATEGORIES = ["all", "preference", "note", "fact", "project", "auto-learned"];

export function MemoryPanel() {
  const { memories, memoriesLoading, setMemories, setMemoriesLoading, removeMemory } =
    useStore();

  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [statsText, setStatsText] = useState("");

  const load = async () => {
    setMemoriesLoading(true);
    try {
      const res = await getMemories(query, category === "all" ? "" : category);
      setMemories(res.memories ?? []);
    } catch (err) {
      console.error(err);
    } finally {
      setMemoriesLoading(false);
    }
  };

  const loadStats = async () => {
    try {
      const res = await getMemoryScores();
      setStatsText(res.result ?? "");
    } catch {
      /* ignore */
    }
  };

  useEffect(() => {
    load();
    loadStats();
  }, []);

  const handleDelete = async (id: number) => {
    await deleteMemory(id);
    removeMemory(id);
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    load();
  };

  const handlePrune = async () => {
    await invokeTool("memory_prune");
    load();
    loadStats();
  };

  return (
    <div className="panel-wrap">
      <div className="panel-header">
        <Brain size={18} />
        <h2>Memory</h2>
        <div className="panel-header-actions">
          <button className="btn-small" onClick={handlePrune} title="Prune low-score memories">
            Prune
          </button>
          <button className="btn-icon" onClick={() => { load(); loadStats(); }}>
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      {/* Stats */}
      {statsText && (
        <div className="memory-stats-box">
          <pre className="stats-pre">{statsText}</pre>
        </div>
      )}

      {/* Search + filter */}
      <form className="search-bar" onSubmit={handleSearch}>
        <Search size={14} className="search-icon" />
        <input
          className="search-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search memories…"
        />
        <button type="submit" className="btn-small">Search</button>
      </form>

      <div className="category-tabs">
        {CATEGORIES.map((c) => (
          <button
            key={c}
            className={`cat-tab ${category === c ? "cat-tab-active" : ""}`}
            onClick={() => {
              setCategory(c);
              setTimeout(load, 0);
            }}
          >
            {c}
          </button>
        ))}
      </div>

      {/* Cards */}
      <div className="memory-list">
        {memoriesLoading ? (
          <div className="panel-loading">Loading…</div>
        ) : memories.length === 0 ? (
          <div className="panel-empty">No memories found.</div>
        ) : (
          memories.map((m) => (
            <MemoryCard
              key={m.id}
              memory={m}
              onDelete={() => handleDelete(m.id)}
            />
          ))
        )}
      </div>
    </div>
  );
}
