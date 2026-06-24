/**
 * src/components/TestPanel.tsx
 * Automated test runner for Tzar — sends queries to /chat and checks responses.
 */
import React, { useState, useRef, useCallback } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────

interface TestCase {
  id: string;
  query: string;
  expect: string[];
  desc: string;
}

interface TestGroup {
  id: string;
  label: string;
  color: string;
  tests: TestCase[];
}

interface TestResult {
  status: "pass" | "fail" | "error" | "aborted" | "running" | "pending";
  response: string;
  elapsed: number;
}

type Results = Record<string, TestResult>;

// ─── Test definitions ─────────────────────────────────────────────────────────

const TEST_GROUPS: TestGroup[] = [
  {
    id: "chat", label: "Basic Chat", color: "#6366f1",
    tests: [
      { id: "chat_1", query: "Hello",                                                           expect: ["hello","hi","assist","help"],       desc: "Greeting" },
      { id: "chat_2", query: "What is a transformer model?",                                    expect: ["transformer","attention","model"],  desc: "Knowledge Q&A" },
      { id: "chat_3", query: "What is 347 divided by 13?",                                      expect: ["26"],                              desc: "Math" },
      { id: "chat_4", query: "Explain the difference between RAG and fine-tuning.",             expect: ["rag","fine","retrieval","train"],   desc: "Concept explanation" },
      { id: "chat_5", query: "Who created Python?",                                             expect: ["guido","rossum"],                  desc: "Factual Q&A" },
    ],
  },
  {
    id: "memory", label: "Memory", color: "#8b5cf6",
    tests: [
      { id: "mem_1", query: "Remember that I prefer responses in bullet points.",               expect: ["got it","saved","noted","sure","preference","remember"],  desc: "Save preference" },
      { id: "mem_2", query: "Remember that my main project is Tzar and I use Ollama locally.", expect: ["got it","saved","noted","sure","remember"],               desc: "Save project fact" },
      { id: "mem_3", query: "My name is Tzar and I'm building a local AI assistant.",          expect: ["got it","saved","noted","tzar","sure"],                   desc: "Save identity" },
      { id: "mem_4", query: "What do you know about me?",                                      expect: ["tzar","project","prefer","ollama","name"],                desc: "Recall facts" },
      { id: "mem_5", query: "Do you remember what my main project is?",                        expect: ["tzar"],                                                   desc: "Recall specific fact" },
      { id: "mem_6", query: "What are my preferences?",                                        expect: ["bullet","prefer"],                                        desc: "Recall preferences" },
      { id: "mem_7", query: "What did we talk about earlier?",                                 expect: ["transformer","python","rag","earlier","discussed"],       desc: "Recall session" },
    ],
  },
  {
    id: "notes", label: "Notes / Obsidian", color: "#0ea5e9",
    tests: [
      { id: "notes_0", query: "Good morning",                                                  expect: ["morning","today","daily","briefing","good"],              desc: "Morning briefing" },
      { id: "notes_1", query: "Create a note called \"Tzar Architecture\" with content: Tzar is a local voice assistant built with FastAPI, Obsidian, and Ollama.", expect: ["creat","note","done","success","architecture"], desc: "Create note" },
      { id: "notes_2", query: "Create a note called \"LLM Comparison\" with content: Qwen 2.5 is good for coding. Llama 3.2 is better for chat.",                  expect: ["creat","note","done","success","comparison"],  desc: "Create 2nd note" },
      { id: "notes_3", query: "Capture this idea: Use a critic layer to evaluate tool results before returning to the user.",                                       expect: ["idea","capture","creat","done","saved"],       desc: "Capture idea" },
      { id: "notes_4", query: "Add to daily note: Worked on Tzar's knowledge graph triple extraction today.",                                                       expect: ["daily","add","append","done","success"],       desc: "Append to daily" },
      { id: "notes_5", query: "Search my notes for knowledge graph.",                                                                                               expect: ["knowledge","graph","note","found","result"],   desc: "Search notes" },
      { id: "notes_6", query: "Search my notes for the word Ollama.",                                                                                               expect: ["ollama","found","note","result"],              desc: "Keyword search" },
      { id: "notes_7", query: "List all my notes.",                                                                                                                 expect: [".md","note","vault","file"],                   desc: "List vault" },
      { id: "notes_8", query: "Read the note \"Tzar Architecture\".",                                                                                               expect: ["fastapi","obsidian","ollama","voice"],         desc: "Read note" },
      { id: "notes_9", query: "What notes are related to LLM?",                                                                                                     expect: ["llm","note","comparison","related"],           desc: "Related notes" },
    ],
  },
  {
    id: "kg", label: "Knowledge Graph", color: "#10b981",
    tests: [
      { id: "kg_1", query: "Show me the knowledge graph summary.",                             expect: ["node","edge","graph","cluster","entity"],                 desc: "KG summary" },
      { id: "kg_2", query: "Add to the knowledge graph: Tzar uses Ollama, Ollama runs Qwen, Tzar is built by Zohaib.", expect: ["graph","added","updated","node","done"], desc: "KG add" },
      { id: "kg_3", query: "Extract knowledge from this text and add to graph: FastAPI is a Python web framework. Tzar uses FastAPI for its backend API.", expect: ["graph","extracted","added","node"], desc: "KG extract" },
      { id: "kg_4", query: "Find the path from Tzar to Python in the knowledge graph.",        expect: ["tzar","python","path","no path"],                         desc: "KG path" },
      { id: "kg_5", query: "Show me the neighbours of Ollama in the knowledge graph.",         expect: ["ollama","qwen","tzar","node","connect"],                  desc: "KG neighbors" },
      { id: "kg_6", query: "Find knowledge graph clusters.",                                   expect: ["cluster","group","node"],                                 desc: "KG clusters" },
      { id: "kg_7", query: "Show me the knowledge graph timeline.",                            expect: ["timeline","node","added","recent","graph"],               desc: "KG timeline" },
      { id: "kg_8", query: "Find orphan nodes in the knowledge graph.",                        expect: ["orphan","node","isolated","no edges","found"],            desc: "KG orphans" },
    ],
  },
  {
    id: "files", label: "File System", color: "#f59e0b",
    tests: [
      { id: "file_1", query: "List all files in my workspace.",                                expect: ["file","dir","workspace","/"],                             desc: "List directory" },
      { id: "file_2", query: "Write a file called test_note.txt with content: This is a test file created by Tzar.", expect: ["written","creat","saved","done"], desc: "Write file" },
      { id: "file_3", query: "Read the file test_note.txt.",                                   expect: ["test file","tzar","created"],                             desc: "Read file" },
      { id: "file_4", query: "Append to the file test_note.txt: Added a second line.",        expect: ["append","added","done","success"],                        desc: "Append file" },
      { id: "file_5", query: "Delete the file test_note.txt.",                                 expect: ["delet","removed","done","success"],                       desc: "Delete file" },
    ],
  },
  {
    id: "web", label: "Web Search", color: "#f43f5e",
    tests: [
      { id: "web_0", query: "Enable web search.",                                              expect: ["enabled","web search","online","done"],                   desc: "Enable" },
      { id: "web_1", query: "Search the web for the latest Ollama release.",                  expect: ["ollama","release","version"],                             desc: "Web search" },
      { id: "web_2", query: "What is the latest version of llama.cpp?",                       expect: ["llama","version","release","cpp"],                        desc: "Research" },
      { id: "web_3", query: "Search online for open source local TTS models 2025.",           expect: ["tts","model","2025"],                                     desc: "Online search" },
      { id: "web_4", query: "Disable web search.",                                             expect: ["disabled","offline","done"],                              desc: "Disable" },
    ],
  },
  {
    id: "projects", label: "Projects", color: "#a855f7",
    tests: [
      { id: "proj_1", query: "List my projects.",                                              expect: ["project","no projects","list"],                           desc: "List projects" },
      { id: "proj_2", query: "Create a new project called \"Tzar Phase 2\" with description: Long-horizon planning.", expect: ["project","creat","done"],      desc: "Create project" },
      { id: "proj_3", query: "Switch to the project Tzar Phase 2.",                           expect: ["switch","tzar phase 2","active","done"],                  desc: "Switch project" },
      { id: "proj_4", query: "What is the status of the current project?",                    expect: ["tzar phase 2","status","project","active"],               desc: "Project status" },
      { id: "proj_5", query: "Archive the project Tzar Phase 2.",                             expect: ["archive","done","success"],                               desc: "Archive project" },
    ],
  },
  {
    id: "system", label: "System", color: "#64748b",
    tests: [
      { id: "sys_1", query: "Show me my memory scores.",                                       expect: ["memory","score","importance","recency"],                  desc: "Memory scores" },
      { id: "sys_2", query: "Prune low-score memories.",                                       expect: ["prune","delet","memor","done","removed"],                 desc: "Prune memories" },
      { id: "sys_3", query: "Show skill learning statistics.",                                 expect: ["skill","pattern","sequence","tool","stat"],               desc: "Skill stats" },
      { id: "sys_4", query: "Show my system profile.",                                         expect: ["cpu","ram","gpu","system","model","hardware"],            desc: "System profile" },
      { id: "sys_5", query: "What model are you using?",                                       expect: ["qwen","llama","model","gguf","ollama"],                   desc: "Model info" },
      { id: "sys_6", query: "What is the MCP server status?",                                  expect: ["mcp","server","connect","no mcp","status"],              desc: "MCP status" },
    ],
  },
  {
    id: "complex", label: "Multi-Label / Complex", color: "#ec4899",
    tests: [
      { id: "cx_1", query: "Create a note about vector databases and add it to the knowledge graph.",                                   expect: ["note","graph","creat","added","done"],        desc: "Tool+Planning" },
      { id: "cx_2", query: "Search my notes for Ollama then create a summary note called \"Ollama Summary\".",                          expect: ["ollama","note","creat","summary","done"],     desc: "Tool chain" },
      { id: "cx_3", query: "Remember that I prefer Urdu for casual chat, then create a note about this preference.",                    expect: ["note","urdu","prefer","creat","done"],        desc: "Memory+Tool" },
    ],
  },
  {
    id: "goals", label: "Goal Tracking", color: "#f97316",
    tests: [
      { id: "goal_1", query: "Build a research summary on local LLM inference: search the web, create a note with findings, add key concepts to the knowledge graph, and log it to my daily note.", expect: ["done","note","graph","daily","created","complet"], desc: "4-step goal" },
      { id: "goal_2", query: "Continue.",                                                       expect: ["goal","step","done","complet","continu","next"],           desc: "Goal continuation" },
      { id: "goal_3", query: "What are my active goals?",                                      expect: ["goal","active","no active","progress","step"],            desc: "List active goals" },
      { id: "goal_4", query: "Show me the status of my current goal.",                         expect: ["goal","step","done","progress","status"],                 desc: "Goal status" },
      { id: "goal_5", query: "Abandon the current goal.",                                      expect: ["abandon","goal","done"],                                  desc: "Abandon goal" },
    ],
  },
  {
    id: "regression", label: "Regression", color: "#475569",
    tests: [
      { id: "reg_1", query: "Hello",                                                           expect: ["hello","hi","assist","help"],                             desc: "Greeting" },
      { id: "reg_2", query: "Remember that I use Arch Linux.",                                 expect: ["got it","saved","noted","arch","sure"],                   desc: "Memory save" },
      { id: "reg_3", query: "Create a note called \"Test\" with content: Regression test.",   expect: ["creat","note","test","done"],                             desc: "Note creation" },
      { id: "reg_4", query: "List my projects.",                                               expect: ["project","list"],                                         desc: "Project list" },
      { id: "reg_5", query: "Search my notes for transformer.",                                expect: ["transformer","note","found","result"],                    desc: "Note search" },
    ],
  },
];

// ─── Test runner ──────────────────────────────────────────────────────────────

async function runTest(test: TestCase, signal: AbortSignal): Promise<TestResult> {
  const start = Date.now();
  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: test.query }),
      signal,
    });
    const elapsed = Date.now() - start;
    if (!res.ok) return { status: "error", response: `HTTP ${res.status}`, elapsed };
    const data = await res.json();
    const response: string = data.response ?? "";
    const lower = response.toLowerCase();
    const passed = test.expect.some(kw => lower.includes(kw.toLowerCase()));
    return { status: passed ? "pass" : "fail", response, elapsed };
  } catch (err: unknown) {
    if ((err as Error).name === "AbortError") return { status: "aborted", response: "Aborted", elapsed: 0 };
    return { status: "error", response: String(err), elapsed: Date.now() - start };
  }
}

// ─── Style helpers ────────────────────────────────────────────────────────────

const STATUS_COLOR: Record<string, string> = {
  pass: "#10b981", fail: "#f43f5e", error: "#f97316",
  aborted: "#64748b", pending: "#334155", running: "#6366f1",
};
const STATUS_ICON: Record<string, string> = {
  pass: "✓", fail: "✗", error: "!", aborted: "–", pending: "·", running: "▶",
};

// ─── Sub-components ───────────────────────────────────────────────────────────

function Badge({ status }: { status: string }) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 3,
      padding: "2px 7px", borderRadius: 4, fontSize: 10, fontWeight: 700,
      background: STATUS_COLOR[status] + "22",
      color: STATUS_COLOR[status],
      border: `1px solid ${STATUS_COLOR[status]}44`,
      whiteSpace: "nowrap",
    }}>
      {STATUS_ICON[status]} {status.toUpperCase()}
    </span>
  );
}

function TestRow({
  test, result, groupColor, onRun,
}: {
  test: TestCase;
  result?: TestResult;
  groupColor: string;
  onRun: (t: TestCase) => void;
}) {
  const [open, setOpen] = useState(false);
  const status = result?.status ?? "pending";

  return (
    <div style={{ borderBottom: "1px solid #1e293b", background: status === "running" ? "#111827" : "transparent" }}>
      <div
        style={{
          display: "grid", gridTemplateColumns: "20px 1fr 90px 60px 30px",
          alignItems: "center", gap: 8, padding: "7px 12px", cursor: result ? "pointer" : "default",
        }}
        onClick={() => result && setOpen(o => !o)}
      >
        <span style={{ color: STATUS_COLOR[status], fontWeight: 700, fontSize: 12 }}>
          {STATUS_ICON[status]}
        </span>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 1 }}>{test.desc}</div>
          <div style={{ fontSize: 10, color: "#475569", fontFamily: "monospace", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {test.query.length > 72 ? test.query.slice(0, 72) + "…" : test.query}
          </div>
        </div>
        <Badge status={status} />
        <div style={{ fontSize: 10, color: "#475569", textAlign: "right" }}>
          {result?.elapsed ? `${result.elapsed}ms` : "—"}
        </div>
        <button
          onClick={e => { e.stopPropagation(); onRun(test); }}
          style={{
            background: groupColor + "22", border: `1px solid ${groupColor}44`,
            color: groupColor, borderRadius: 3, padding: "2px 5px",
            fontSize: 10, cursor: "pointer", fontWeight: 700, lineHeight: 1,
          }}
        >↺</button>
      </div>
      {open && result && (
        <div style={{
          margin: "0 12px 8px", padding: 10, background: "#0a0f1a",
          borderRadius: 5, border: "1px solid #1e293b",
        }}>
          <div style={{ fontSize: 10, color: "#475569", marginBottom: 4 }}>
            Expected any of: {test.expect.join(", ")}
          </div>
          <div style={{ fontSize: 11, color: result.status === "pass" ? "#86efac" : "#fca5a5", fontFamily: "monospace", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
            {result.response || "(empty response)"}
          </div>
        </div>
      )}
    </div>
  );
}

function GroupCard({
  group, results, running, onRunGroup, onRunTest,
}: {
  group: TestGroup;
  results: Results;
  running: boolean;
  onRunGroup: (g: TestGroup) => void;
  onRunTest: (t: TestCase) => void;
}) {
  const [collapsed, setCollapsed] = useState(false);
  const groupResults = group.tests.map(t => results[t.id]).filter(Boolean) as TestResult[];
  const passed = groupResults.filter(r => r.status === "pass").length;
  const failed = groupResults.filter(r => r.status === "fail").length;
  const errors = groupResults.filter(r => r.status === "error").length;
  const pct = groupResults.length > 0 ? Math.round((passed / groupResults.length) * 100) : 0;

  return (
    <div style={{ border: `1px solid ${group.color}33`, borderRadius: 8, marginBottom: 10, overflow: "hidden" }}>
      <div
        style={{
          display: "flex", alignItems: "center", gap: 8,
          padding: "9px 12px", background: group.color + "0d", cursor: "pointer",
        }}
        onClick={() => setCollapsed(c => !c)}
      >
        <span style={{ color: group.color, fontSize: 11 }}>{collapsed ? "▶" : "▼"}</span>
        <span style={{ fontWeight: 700, color: "#e2e8f0", fontSize: 13, flex: 1 }}>{group.label}</span>
        <span style={{ fontSize: 10, color: "#475569" }}>{groupResults.length}/{group.tests.length}</span>
        {groupResults.length > 0 && (
          <>
            <span style={{ fontSize: 10, color: "#10b981" }}>✓{passed}</span>
            {failed > 0 && <span style={{ fontSize: 10, color: "#f43f5e" }}>✗{failed}</span>}
            {errors > 0 && <span style={{ fontSize: 10, color: "#f97316" }}>!{errors}</span>}
            <div style={{ width: 50, height: 5, background: "#1e293b", borderRadius: 3, overflow: "hidden" }}>
              <div style={{
                width: `${pct}%`, height: "100%",
                background: pct === 100 ? "#10b981" : group.color,
                transition: "width 0.3s",
              }} />
            </div>
          </>
        )}
        <button
          onClick={e => { e.stopPropagation(); onRunGroup(group); }}
          disabled={running}
          style={{
            background: group.color + "22", border: `1px solid ${group.color}55`,
            color: group.color, borderRadius: 4, padding: "3px 9px",
            fontSize: 10, fontWeight: 700, cursor: running ? "not-allowed" : "pointer",
          }}
        >
          Run
        </button>
      </div>
      {!collapsed && group.tests.map(test => (
        <TestRow
          key={test.id}
          test={test}
          result={results[test.id]}
          groupColor={group.color}
          onRun={onRunTest}
        />
      ))}
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export function TestPanel() {
  const [results, setResults]   = useState<Results>({});
  const [running, setRunning]   = useState(false);
  const [delay, setDelay]       = useState(1500);
  const [filter, setFilter]     = useState("all");
  const [log, setLog]           = useState<{ msg: string; color: string }[]>([]);
  const abortRef                = useRef<AbortController | null>(null);
  const logRef                  = useRef<HTMLDivElement>(null);

  const addLog = useCallback((msg: string, color = "#64748b") => {
    setLog(l => [...l.slice(-300), { msg, color }]);
    setTimeout(() => logRef.current?.scrollTo(0, logRef.current.scrollHeight), 30);
  }, []);

  const setResult = useCallback((id: string, result: TestResult) => {
    setResults(r => ({ ...r, [id]: result }));
  }, []);

  const runSingle = useCallback(async (test: TestCase) => {
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    addLog(`▶ ${test.desc}`, "#6366f1");
    setResult(test.id, { status: "running", response: "", elapsed: 0 });
    const result = await runTest(test, ctrl.signal);
    setResult(test.id, result);
    const c = result.status === "pass" ? "#10b981" : result.status === "fail" ? "#f43f5e" : "#f97316";
    addLog(`  ${STATUS_ICON[result.status]} ${result.status} (${result.elapsed}ms) ${result.response.slice(0, 80)}`, c);
    return result;
  }, [addLog, setResult]);

  const runGroup = useCallback(async (group: TestGroup) => {
    if (running) return;
    setRunning(true);
    abortRef.current = new AbortController();
    addLog(`\n── ${group.label} ──`, group.color);
    for (const test of group.tests) {
      if (abortRef.current?.signal.aborted) break;
      await runSingle(test);
      await new Promise(r => setTimeout(r, delay));
    }
    setRunning(false);
  }, [running, runSingle, delay, addLog]);

  const runAll = useCallback(async () => {
    if (running) return;
    const total = TEST_GROUPS.reduce((s, g) => s + g.tests.length, 0);
    setRunning(true);
    abortRef.current = new AbortController();
    addLog(`TZAR TEST SUITE — ${total} tests`, "#e2e8f0");
    for (const group of TEST_GROUPS) {
      if (abortRef.current?.signal.aborted) break;
      addLog(`\n── ${group.label} ──`, group.color);
      for (const test of group.tests) {
        if (abortRef.current?.signal.aborted) break;
        await runSingle(test);
        await new Promise(r => setTimeout(r, delay));
      }
    }
    const all = Object.values(results);
    const pass = all.filter(r => r.status === "pass").length;
    addLog(`\nDONE — ${pass}/${all.length} passed`, pass === all.length ? "#10b981" : "#f43f5e");
    setRunning(false);
  }, [running, runSingle, delay, addLog, results]);

  const abort = useCallback(() => {
    abortRef.current?.abort();
    setRunning(false);
    addLog("⏹ Stopped", "#f43f5e");
  }, [addLog]);

  // Stats
  const allR    = Object.values(results);
  const total   = TEST_GROUPS.reduce((s, g) => s + g.tests.length, 0);
  const passed  = allR.filter(r => r.status === "pass").length;
  const failed  = allR.filter(r => r.status === "fail").length;
  const errors  = allR.filter(r => r.status === "error").length;
  const pct     = allR.length > 0 ? Math.round((passed / allR.length) * 100) : 0;

  const filtered = filter === "all"
    ? TEST_GROUPS
    : TEST_GROUPS.map(g => ({
        ...g,
        tests: g.tests.filter(t => {
          const r = results[t.id];
          if (!r) return filter === "pending";
          return r.status === filter;
        }),
      })).filter(g => g.tests.length > 0);

  return (
    <div style={{
      height: "100%", display: "grid",
      gridTemplateColumns: "1fr 320px",
      gridTemplateRows: "auto 1fr",
      background: "#0a0f1a", color: "#e2e8f0",
      fontFamily: "'JetBrains Mono','Fira Code',monospace",
      overflow: "hidden",
    }}>

      {/* ── Header ── */}
      <div style={{
        gridColumn: "1 / -1", display: "flex", alignItems: "center", gap: 12,
        padding: "10px 16px", background: "#0d1526", borderBottom: "1px solid #1e293b",
        flexWrap: "wrap",
      }}>
        <div>
          <div style={{ fontWeight: 800, fontSize: 14 }}>TZAR — Test Runner</div>
          <div style={{ fontSize: 10, color: "#475569" }}>{total} tests · {TEST_GROUPS.length} groups</div>
        </div>

        <div style={{ flex: 1 }} />

        {/* Progress */}
        {allR.length > 0 && (
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 100, height: 6, background: "#1e293b", borderRadius: 3, overflow: "hidden" }}>
              <div style={{ width: `${pct}%`, height: "100%", background: pct === 100 ? "#10b981" : "#6366f1", transition: "width 0.3s" }} />
            </div>
            <span style={{ fontSize: 11 }}>{pct}%</span>
            <span style={{ fontSize: 11, color: "#10b981" }}>✓{passed}</span>
            <span style={{ fontSize: 11, color: "#f43f5e" }}>✗{failed}</span>
            {errors > 0 && <span style={{ fontSize: 11, color: "#f97316" }}>!{errors}</span>}
          </div>
        )}

        {/* Delay */}
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span style={{ fontSize: 10, color: "#64748b" }}>delay</span>
          <select value={delay} onChange={e => setDelay(Number(e.target.value))} style={{
            background: "#1e293b", border: "1px solid #334155", color: "#e2e8f0",
            borderRadius: 4, padding: "3px 5px", fontSize: 10,
          }}>
            {[500, 1000, 1500, 2000, 3000].map(d => <option key={d} value={d}>{d}ms</option>)}
          </select>
        </div>

        {/* Buttons */}
        <button onClick={runAll} disabled={running} style={{
          background: running ? "#1e293b" : "#6366f1", color: "#fff",
          border: "none", borderRadius: 5, padding: "6px 14px",
          fontWeight: 700, fontSize: 12, cursor: running ? "not-allowed" : "pointer",
        }}>
          {running ? "Running…" : "▶ Run All"}
        </button>
        {running && (
          <button onClick={abort} style={{
            background: "#7f1d1d", color: "#fca5a5", border: "none",
            borderRadius: 5, padding: "6px 10px", fontSize: 12, cursor: "pointer", fontWeight: 700,
          }}>⏹ Stop</button>
        )}
        <button onClick={() => { setResults({}); setLog([]); }} disabled={running} style={{
          background: "#1e293b", color: "#94a3b8",
          border: "1px solid #334155", borderRadius: 5, padding: "6px 10px",
          fontSize: 12, cursor: "pointer",
        }}>Clear</button>
      </div>

      {/* ── Left: test groups ── */}
      <div style={{ overflow: "auto", padding: 14 }}>
        {/* Filter tabs */}
        <div style={{ display: "flex", gap: 5, marginBottom: 12, flexWrap: "wrap" }}>
          {["all","pass","fail","error","pending"].map(f => (
            <button key={f} onClick={() => setFilter(f)} style={{
              padding: "3px 10px", borderRadius: 4, fontSize: 10, fontWeight: 600,
              cursor: "pointer",
              background: filter === f ? STATUS_COLOR[f] + "33" : "#1e293b",
              color: filter === f ? STATUS_COLOR[f] : "#64748b",
              border: `1px solid ${filter === f ? STATUS_COLOR[f] + "55" : "#334155"}`,
            }}>
              {f === "all" ? `All (${total})` : `${STATUS_ICON[f] ?? "·"} ${f}`}
            </button>
          ))}
        </div>

        {filtered.map(group => (
          <GroupCard
            key={group.id}
            group={group}
            results={results}
            running={running}
            onRunGroup={runGroup}
            onRunTest={runSingle}
          />
        ))}
      </div>

      {/* ── Right: live log ── */}
      <div style={{ borderLeft: "1px solid #1e293b", display: "flex", flexDirection: "column", background: "#080d18", overflow: "hidden" }}>
        <div style={{ padding: "8px 12px", borderBottom: "1px solid #1e293b", fontSize: 10, color: "#475569", fontWeight: 700, letterSpacing: "0.08em" }}>
          LIVE LOG
        </div>
        <div ref={logRef} style={{ flex: 1, overflow: "auto", padding: 10, fontSize: 10, lineHeight: 1.8 }}>
          {log.length === 0
            ? <div style={{ color: "#1e293b" }}>No output yet…</div>
            : log.map((l, i) => (
              <div key={i} style={{ color: l.color, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>{l.msg}</div>
            ))
          }
        </div>
        <div style={{ padding: 8, borderTop: "1px solid #1e293b", display: "flex", gap: 6 }}>
          <button
            onClick={() => navigator.clipboard?.writeText(log.map(l => l.msg).join("\n"))}
            style={{ flex: 1, background: "#1e293b", border: "1px solid #334155", color: "#64748b", borderRadius: 4, padding: 5, fontSize: 10, cursor: "pointer" }}
          >Copy</button>
          <button
            onClick={() => setLog([])}
            style={{ flex: 1, background: "#1e293b", border: "1px solid #334155", color: "#64748b", borderRadius: 4, padding: 5, fontSize: 10, cursor: "pointer" }}
          >Clear</button>
        </div>
      </div>
    </div>
  );
}