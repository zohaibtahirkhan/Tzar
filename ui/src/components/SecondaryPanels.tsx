/**
 * src/components/SkillsPanel.tsx
 * src/components/ProjectsPanel.tsx
 * src/components/SettingsPanel.tsx
 *
 * All three secondary panels in one file.
 */

import React, { useEffect, useState } from "react";
import {
  Layers, RefreshCw, FolderOpen, Circle,
  Wifi, WifiOff, Server, Settings, ChevronRight,
} from "lucide-react";
import { useStore } from "../stores/useStore";
import {
  invokeTool, getProjectList, switchProject, setWebSearch, getHealth,
} from "../api";

// ═══════════════════════════════════════════════════════════════════════════════
// Skills Panel
// ═══════════════════════════════════════════════════════════════════════════════

export function SkillsPanel() {
  const { skillStats, setSkillStats } = useStore();
  const [loading, setLoading] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const res = await invokeTool("skill_learning_stats");
      setSkillStats(res.result ?? "");
    } catch {
      /* ignore */
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  // Parse the plain-text stats into structured lines
  const lines = skillStats.split("\n").filter(Boolean);
  const headerLine = lines[0] ?? "";
  const restLines = lines.slice(1);

  return (
    <div className="panel-wrap">
      <div className="panel-header">
        <Layers size={18} />
        <h2>Skill Learning</h2>
        <button className="btn-icon" onClick={load}><RefreshCw size={14} /></button>
      </div>

      {loading ? (
        <div className="panel-loading">Loading…</div>
      ) : skillStats ? (
        <div className="skills-content">
          <div className="skills-header-line">{headerLine}</div>
          <div className="skills-lines">
            {restLines.map((line, i) => {
              const isPattern = line.trim().startsWith("note") ||
                                line.trim().startsWith("web") ||
                                line.trim().startsWith("doc") ||
                                line.trim().startsWith("obs") ||
                                line.trim().startsWith("kg");
              return (
                <div
                  key={i}
                  className={`skill-line ${isPattern ? "skill-pattern" : ""}`}
                >
                  {line.trim().startsWith("Top") ? (
                    <span className="skill-section-label">{line}</span>
                  ) : isPattern ? (
                    <div className="skill-pattern-row">
                      <Layers size={11} />
                      <span>{line.trim()}</span>
                    </div>
                  ) : (
                    <span className="skill-meta">{line}</span>
                  )}
                </div>
              );
            })}
          </div>
          <div className="skills-hint">
            Patterns are automatically detected from repeated tool usage.
            At 3 occurrences, Hafiz will ask to save it as a skill.
          </div>
        </div>
      ) : (
        <div className="panel-empty">
          No skill patterns detected yet. Use Hafiz for a while and patterns
          will appear here.
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Projects Panel
// ═══════════════════════════════════════════════════════════════════════════════

interface Project {
  name: string;
  status: string;
  description: string;
  last_opened: string | null;
}

function parseProjectList(raw: string): Project[] {
  const projects: Project[] = [];
  const lines = raw.split("\n").filter((l) => l.trim().startsWith("●") ||
    l.trim().startsWith("◐") || l.trim().startsWith("○"));

  for (const line of lines) {
    const statusChar = line.trim()[0];
    const status =
      statusChar === "●" ? "active" :
      statusChar === "◐" ? "paused" : "archived";
    // Format: "● Name — description (last: date)"
    const m = line.match(/[●◐○]\s+(.+?)(?:\s+—\s+(.+?))?(?:\s+\(last:.+\))?$/);
    if (m) {
      projects.push({
        name: m[1].trim(),
        status,
        description: m[2]?.trim() ?? "",
        last_opened: null,
      });
    }
  }
  return projects;
}

export function ProjectsPanel() {
  const { activeProject, setActiveProject, setPanel, addMessage } = useStore();
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(false);
  const [newName, setNewName] = useState("");

  const load = async () => {
    setLoading(true);
    try {
      const res = await invokeTool("project_list");
      setProjects(parseProjectList(res.result ?? ""));
    } catch {
      /* ignore */
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const handleSwitch = async (name: string) => {
    const res = await switchProject(name);
    setActiveProject(name);
    // Show the context load result in chat
    addMessage({
      id: `proj-${Date.now()}`,
      role: "assistant",
      content: res.result ?? `Switched to project: ${name}`,
      ts: Date.now(),
    });
    setPanel("chat");
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newName.trim()) return;
    await invokeTool("project_new", { name: newName.trim() });
    setNewName("");
    load();
  };

  const statusIcon = (status: string) => {
    if (status === "active") return <Circle size={8} fill="var(--accent-green)" color="var(--accent-green)" />;
    if (status === "paused") return <Circle size={8} fill="var(--accent-amber)" color="var(--accent-amber)" />;
    return <Circle size={8} fill="var(--muted)" color="var(--muted)" />;
  };

  return (
    <div className="panel-wrap">
      <div className="panel-header">
        <FolderOpen size={18} />
        <h2>Projects</h2>
        <button className="btn-icon" onClick={load}><RefreshCw size={14} /></button>
      </div>

      {activeProject && (
        <div className="active-project-banner">
          <Circle size={8} fill="var(--accent-green)" color="var(--accent-green)" />
          Active: <strong>{activeProject}</strong>
        </div>
      )}

      {/* Create new */}
      <form className="new-project-form" onSubmit={handleCreate}>
        <input
          className="search-input"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          placeholder="New project name…"
        />
        <button type="submit" className="btn-small" disabled={!newName.trim()}>
          Create
        </button>
      </form>

      {loading ? (
        <div className="panel-loading">Loading…</div>
      ) : projects.length === 0 ? (
        <div className="panel-empty">No projects yet. Create one above.</div>
      ) : (
        <div className="project-list">
          {projects.map((p) => (
            <div
              key={p.name}
              className={`project-card ${activeProject === p.name ? "project-active" : ""}`}
              onClick={() => handleSwitch(p.name)}
            >
              <div className="project-card-left">
                {statusIcon(p.status)}
                <div>
                  <div className="project-name">{p.name}</div>
                  {p.description && (
                    <div className="project-desc">{p.description}</div>
                  )}
                </div>
              </div>
              <ChevronRight size={14} className="project-arrow" />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Settings Panel
// ═══════════════════════════════════════════════════════════════════════════════

export function SettingsPanel() {
  const { health, setHealth, webSearchEnabled, setWebSearchEnabled } = useStore();
  const [mcpStatus, setMcpStatus] = useState("");
  const [loading, setLoading] = useState(false);

  const loadAll = async () => {
    setLoading(true);
    try {
      const h = await getHealth();
      setHealth(h);
      setWebSearchEnabled(h.web_search);

      const mcp = await invokeTool("mcp_status");
      setMcpStatus(mcp.result ?? "");
    } catch {
      /* ignore */
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadAll(); }, []);

  const handleWebSearch = async (enabled: boolean) => {
    await setWebSearch(enabled);
    setWebSearchEnabled(enabled);
  };

  const statusDot = (val: string) => (
    <span
      className="status-dot"
      style={{
        background: val === "loaded" ? "var(--accent-green)" : "var(--accent-red)",
      }}
    />
  );

  return (
    <div className="panel-wrap">
      <div className="panel-header">
        <Settings size={18} />
        <h2>Settings</h2>
        <button className="btn-icon" onClick={loadAll}><RefreshCw size={14} /></button>
      </div>

      {loading ? (
        <div className="panel-loading">Loading…</div>
      ) : (
        <>
          {/* System status */}
          <div className="settings-section">
            <div className="settings-section-label">System Status</div>
            <div className="status-grid">
              {health && (
                <>
                  <div className="status-row">
                    {statusDot(health.llm)}
                    <span>LLM</span>
                    <span className="status-value">{health.llm}</span>
                  </div>
                  <div className="status-row">
                    {statusDot(health.stt)}
                    <span>STT (Whisper)</span>
                    <span className="status-value">{health.stt}</span>
                  </div>
                  <div className="status-row">
                    {statusDot(health.tts)}
                    <span>TTS (Kokoro)</span>
                    <span className="status-value">{health.tts}</span>
                  </div>
                  <div className="status-row">
                    <span className="status-dot" style={{ background: "var(--muted)" }} />
                    <span>Workspace</span>
                    <span className="status-value monospace">{health.workspace}</span>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Web search toggle */}
          <div className="settings-section">
            <div className="settings-section-label">Features</div>
            <div className="settings-row">
              <div>
                <div className="settings-row-label">Web Search</div>
                <div className="settings-row-sub">
                  Allow Hafiz to search the web (DuckDuckGo)
                </div>
              </div>
              <button
                className={`toggle-btn ${webSearchEnabled ? "toggle-on" : ""}`}
                onClick={() => handleWebSearch(!webSearchEnabled)}
              >
                {webSearchEnabled ? (
                  <><Wifi size={13} /> On</>
                ) : (
                  <><WifiOff size={13} /> Off</>
                )}
              </button>
            </div>
          </div>

          {/* MCP status */}
          <div className="settings-section">
            <div className="settings-section-label">
              <Server size={13} /> MCP Servers
            </div>
            {mcpStatus ? (
              <pre className="stats-pre mcp-pre">{mcpStatus}</pre>
            ) : (
              <div className="panel-empty small">
                No MCP servers configured. Set MCP_SERVERS in your .env file.
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
