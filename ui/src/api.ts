/**
 * src/api.ts — complete typed client for the Tzar FastAPI backend.
 */

const BASE = "http://127.0.0.1:8000";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  llm: string;
  stt: string;
  tts: string;
  wake_word: string;
  web_search: boolean;
  multi_agent: boolean;
  workspace: string;
}

export interface ChatResponse {
  response: string;
  tool_results?: ToolResult[];
}

export interface ToolResult {
  tool: string;
  status: string;
  result: string;
}

export interface Memory {
  id: number;
  category: string;
  content: string;
  created_at?: string;
  importance?: number;
  recency?: number;
  frequency?: number;
  composite?: number;
}

export interface MemoryScoreResponse {
  memories: Memory[];
}

export interface RagDoc {
  title: string;
  source_path: string;
  chunks: number;
  last_indexed: string;
}

// ─── Core fetch ───────────────────────────────────────────────────────────────

async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API ${path} ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ─── Health ───────────────────────────────────────────────────────────────────

export const getHealth = () => apiFetch<HealthResponse>("/health");

// ─── Chat ─────────────────────────────────────────────────────────────────────

export const sendChat = (message: string) =>
  apiFetch<ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify({ message }),
  });

export function streamChat(
  message: string,
  onToken: (token: string) => void,
  onToolResults: (results: ToolResult[]) => void,
  onDone: () => void,
  onError: (err: string) => void
): AbortController {
  const ctrl = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${BASE}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
        signal: ctrl.signal,
      });

      if (!res.ok || !res.body) { onError(`Stream error: ${res.status}`); return; }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6).trim();
          if (data === "[DONE]") { onDone(); return; }
          try {
            const parsed = JSON.parse(data);
            if (parsed.token)        onToken(parsed.token);
            if (parsed.tool_results) onToolResults(parsed.tool_results);
            if (parsed.error)        onError(parsed.error);
          } catch { /* ignore */ }
        }
      }
      onDone();
    } catch (err: unknown) {
      if ((err as Error).name !== "AbortError") onError(String(err));
    }
  })();

  return ctrl;
}

// ─── Memory ───────────────────────────────────────────────────────────────────

export const getMemories = (q = "", category = "") =>
  apiFetch<MemoryScoreResponse>(
    `/memory?q=${encodeURIComponent(q)}&category=${encodeURIComponent(category)}`
  );

export const saveMemory = (category: string, content: string) =>
  apiFetch<{ id: number; status: string }>("/memory", {
    method: "POST",
    body: JSON.stringify({ category, content }),
  });

export const deleteMemory = (id: number) =>
  apiFetch<{ deleted: boolean }>(`/memory/${id}`, { method: "DELETE" });

// ─── Tool ─────────────────────────────────────────────────────────────────────

export const invokeTool = (tool: string, params: Record<string, unknown> = {}) =>
  apiFetch<ToolResult>("/tool", {
    method: "POST",
    body: JSON.stringify({ tool, params }),
  });

// ─── Settings ─────────────────────────────────────────────────────────────────

export const setWebSearch = (enabled: boolean) =>
  apiFetch<{ web_search: boolean; status: string }>("/settings/web-search", {
    method: "POST",
    body: JSON.stringify({ enabled }),
  });

export const setMultiAgent = (enabled: boolean) =>
  apiFetch<{ multi_agent: boolean; status: string }>("/settings/multi-agent", {
    method: "POST",
    body: JSON.stringify({ enabled }),
  });

// ─── Named tool wrappers ──────────────────────────────────────────────────────

export const getKGSummary      = () => invokeTool("kg_summary");
export const getKGNeighbors    = (node: string, depth = 1) => invokeTool("kg_neighbors", { node, depth });
export const getSkillStats     = () => invokeTool("skill_learning_stats");
export const getMemoryScores   = () => invokeTool("memory_scores");
export const getProjectList    = () => invokeTool("project_list");
export const switchProject     = (name: string) => invokeTool("project_switch", { name });
export const getMCPStatus      = () => invokeTool("mcp_status");
export const getSystemProfile  = () => invokeTool("system_profile");
export const listDocuments     = () => invokeTool("list_documents");
export const ingestDocument    = (path: string) => invokeTool("ingest_document", { path });
export const ingestDirectory   = (directory: string) => invokeTool("ingest_directory", { directory });
export const removeDocument    = (path: string) => invokeTool("remove_document", { path });
export const docSearch         = (query: string) => invokeTool("doc_search", { query });
export const unifiedSearch     = (query: string) => invokeTool("unified_search", { query });
export const pruneMemories     = () => invokeTool("memory_prune");
export const getProjectStatus  = () => invokeTool("project_status");