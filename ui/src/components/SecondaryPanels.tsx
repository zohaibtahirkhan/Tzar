/**
 * src/components/SecondaryPanels.tsx
 * Skills, Projects, Settings panels.
 */
import React, { useEffect, useState } from "react";
import {
  Layers, RefreshCw, FolderOpen, Circle,
  Wifi, WifiOff, Server, Settings, ChevronRight, Plus, Radio,
} from "lucide-react";
import { useStore } from "../stores/useStore";
import { invokeTool, switchProject, setWebSearch, setMultiAgent, getHealth } from "../api";

// ═══════════════════════════════════════════════════════════════════════════════
// Skills Panel
// ═══════════════════════════════════════════════════════════════════════════════

export function SkillsPanel() {
  const { skillStats, setSkillStats } = useStore();
  const [loading, setLoading] = useState(false);

  const load = async () => {
    setLoading(true);
    try { const r = await invokeTool("skill_learning_stats"); setSkillStats(r.result ?? ""); }
    catch { /* ignore */ } finally { setLoading(false); }
  };

  useEffect(() => { load(); }, []);

  const lines = skillStats.split("\n").filter(Boolean);

  return (
    <div className="panel-wrap">
      <div className="panel-header">
        <Layers size={17} />
        <h2>Skill Learning</h2>
        <button className="btn-icon" onClick={load}><RefreshCw size={13} /></button>
      </div>

      {loading ? <div className="panel-loading">Loading…</div>
      : skillStats ? (
        <div className="skills-body">
          <div className="skills-headline">{lines[0]}</div>
          {lines.slice(1).map((line, i) => {
            const isSection = line.trim().startsWith("Top") || line.trim().startsWith("Await");
            const isPattern = /^\s+([\w-]+.*?)\(seen/.test(line);
            return (
              <div key={i} className={`skill-line ${isSection ? "skill-section" : isPattern ? "skill-pattern-line" : "skill-meta-line"}`}>
                {isPattern
                  ? <div className="skill-pattern-chip"><Layers size={10} /><span>{line.trim()}</span></div>
                  : <span>{line}</span>
                }
              </div>
            );
          })}
          <div className="skills-hint">
            Tzar detects repeated tool sequences automatically.
            After 3 occurrences it will ask you to save it as a named skill.
          </div>
        </div>
      ) : (
        <div className="panel-empty">No patterns detected yet.</div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Projects Panel
// ═══════════════════════════════════════════════════════════════════════════════

interface Project { name: string; status: string; description: string; }

function parseProjects(raw: string): Project[] {
  return raw.split("\n")
    .filter(l => /[●◐○]/.test(l))
    .map(l => {
      const ch = l.trim()[0];
      const status = ch === "●" ? "active" : ch === "◐" ? "paused" : "archived";
      const m = l.match(/[●◐○]\s+(.+?)(?:\s+—\s+(.+?))?(?:\s+\(last:.+\))?$/);
      return { name: m?.[1]?.trim() ?? "", status, description: m?.[2]?.trim() ?? "" };
    })
    .filter(p => p.name);
}

export function ProjectsPanel() {
  const { activeProject, setActiveProject, setPanel, addMessage } = useStore();
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading]   = useState(false);
  const [newName, setNewName]   = useState("");
  const [newDesc, setNewDesc]   = useState("");
  const [showForm, setShowForm] = useState(false);

  const load = async () => {
    setLoading(true);
    try { const r = await invokeTool("project_list"); setProjects(parseProjects(r.result ?? "")); }
    catch { /* ignore */ } finally { setLoading(false); }
  };

  useEffect(() => { load(); }, []);

  const handleSwitch = async (name: string) => {
    const res = await switchProject(name);
    setActiveProject(name);
    addMessage({ id: `proj-${Date.now()}`, role: "assistant", content: res.result ?? `Switched to ${name}`, ts: Date.now() });
    setPanel("chat");
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newName.trim()) return;
    await invokeTool("project_new", { name: newName.trim(), description: newDesc.trim() });
    setNewName(""); setNewDesc(""); setShowForm(false);
    load();
  };

  const dot = (status: string) => {
    const c = status === "active" ? "var(--accent-green)" : status === "paused" ? "var(--accent-amber)" : "var(--muted)";
    return <Circle size={8} fill={c} color={c} />;
  };

  return (
    <div className="panel-wrap">
      <div className="panel-header">
        <FolderOpen size={17} />
        <h2>Projects</h2>
        <div className="panel-header-actions">
          <button className="btn-small" onClick={() => setShowForm(s => !s)}>
            <Plus size={11} /> New
          </button>
          <button className="btn-icon" onClick={load}><RefreshCw size={13} /></button>
        </div>
      </div>

      {activeProject && (
        <div className="active-banner">
          <Circle size={7} fill="var(--accent-green)" color="var(--accent-green)" />
          Active: <strong>{activeProject}</strong>
        </div>
      )}

      {showForm && (
        <form className="new-project-form" onSubmit={handleCreate}>
          <input className="search-input" value={newName} onChange={e => setNewName(e.target.value)} placeholder="Project name…" />
          <input
            className="search-input"
            value={newDesc}
            onChange={e => setNewDesc(e.target.value)}
            placeholder="Description (optional)"
            style={{ marginTop: 6 }}
            onKeyDown={e => e.key === "Enter" && e.preventDefault()}
          />
          <button type="submit" className="btn-small" style={{ marginTop: 6 }} disabled={!newName.trim()}>Create</button>
        </form>
      )}

      {loading ? <div className="panel-loading">Loading…</div>
      : projects.length === 0 ? <div className="panel-empty">No projects yet.</div>
      : (
        <div className="project-list">
          {projects.map(p => (
            <div key={p.name} className={`project-card ${activeProject === p.name ? "project-active" : ""}`}
              onClick={() => handleSwitch(p.name)}>
              <div className="project-left">
                {dot(p.status)}
                <div>
                  <div className="project-name">{p.name}</div>
                  {p.description && <div className="project-desc">{p.description}</div>}
                </div>
              </div>
              <ChevronRight size={13} className="project-arrow" />
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
  const { health, setHealth, webSearchEnabled, setWebSearchEnabled, multiAgentEnabled, setMultiAgentEnabled } = useStore();
  const [mcpStatus, setMcpStatus] = useState("");
  const [loading, setLoading]     = useState(false);

  const loadAll = async () => {
    setLoading(true);
    try {
      const h = await getHealth();
      setHealth(h);
      setWebSearchEnabled(h.web_search);
      setMultiAgentEnabled(h.multi_agent);
      const mcp = await invokeTool("mcp_status");
      setMcpStatus(mcp.result ?? "");
    } catch { /* ignore */ } finally { setLoading(false); }
  };

  useEffect(() => { loadAll(); }, []);

  const handleWebSearch = async (v: boolean) => {
    await setWebSearch(v);
    setWebSearchEnabled(v);
  };

  const handleMultiAgent = async (v: boolean) => {
    await setMultiAgent(v);
    setMultiAgentEnabled(v);
  };

  const statusDot = (val: string, goodVal = "loaded") => (
    <span
      className="status-dot"
      style={{ background: val === goodVal ? "var(--green)" : val === "failed" ? "var(--red)" : "var(--amber)" }}
    />
  );

  return (
    <div className="panel-wrap">
      <div className="panel-header">
        <Settings size={17} />
        <h2>Settings</h2>
        <button className="btn-icon" onClick={loadAll}><RefreshCw size={13} /></button>
      </div>

      {loading ? <div className="panel-loading">Loading…</div> : (
        <div className="settings-body">
          {/* System status */}
          <div className="settings-section">
            <div className="settings-label">System Status</div>
            {health ? (
              <div className="status-grid">
                {[
                  ["LLM",            health.llm],
                  ["STT (Whisper)",  health.stt],
                  ["TTS (Kokoro)",   health.tts],
                  ["Wake Word",      health.wake_word],
                ].map(([label, val]) => (
                  <div key={label} className="status-row">
                    {statusDot(val)}
                    <span className="status-name">{label}</span>
                    <span className="status-val">{val}</span>
                  </div>
                ))}
                <div className="status-row">
                  <span className="status-dot" style={{ background: "var(--muted)" }} />
                  <span className="status-name">Workspace</span>
                  <span className="status-val mono">{health.workspace}</span>
                </div>
              </div>
            ) : <div className="panel-empty small">Backend not reachable.</div>}
          </div>

          {/* Feature toggles */}
          <div className="settings-section">
            <div className="settings-label">Features</div>
            <div className="toggle-row">
              <div>
                <div className="toggle-title">Web Search</div>
                <div className="toggle-sub">DuckDuckGo search for real-time information</div>
              </div>
              <button
                className={`toggle-pill ${webSearchEnabled ? "toggle-on" : ""}`}
                onClick={() => handleWebSearch(!webSearchEnabled)}
              >
                {webSearchEnabled ? <><Wifi size={12} /> On</> : <><WifiOff size={12} /> Off</>}
              </button>
            </div>
            <div className="toggle-row" style={{ marginTop: 10 }}>
              <div>
                <div className="toggle-title">Multi-Agent Mode</div>
                <div className="toggle-sub">Parallel Planner / Researcher / Executor agents</div>
              </div>
              <button
                className={`toggle-pill ${multiAgentEnabled ? "toggle-on" : ""}`}
                onClick={() => handleMultiAgent(!multiAgentEnabled)}
              >
                {multiAgentEnabled ? <><Radio size={12} /> On</> : "Off"}
              </button>
            </div>
          </div>

          {/* MCP servers */}
          <div className="settings-section">
            <div className="settings-label"><Server size={12} /> MCP Servers</div>
            {mcpStatus
              ? <pre className="stats-pre mcp-pre">{mcpStatus}</pre>
              : <div className="panel-empty small">No MCP servers configured.<br />Set MCP_SERVERS in your .env.</div>
            }
          </div>
        </div>
      )}
    </div>
  );
}
