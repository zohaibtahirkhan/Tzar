/**
 * src/stores/useStore.ts — global Zustand store
 */
import { create } from "zustand";
import type { Memory, HealthResponse, RagDoc } from "../api";

export type Panel = "chat" | "memory" | "graph" | "skills" | "projects" | "docs" | "system" | "settings";

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  streaming?: boolean;
  toolResults?: { tool: string; result: string; status: string }[];
  ts: number;
}

export interface KGNode { id: string; label: string; group: string; }
export interface KGEdge { source: string; target: string; relation: string; }

let _msgId = 0;
export const nextId = () => `msg-${++_msgId}-${Date.now()}`;

interface State {
  // Nav
  activePanel: Panel;
  setPanel: (p: Panel) => void;

  // Health
  health: HealthResponse | null;
  setHealth: (h: HealthResponse) => void;

  // Chat
  messages: Message[];
  isStreaming: boolean;
  addMessage: (m: Message) => void;
  appendToken: (id: string, token: string) => void;
  finaliseStream: (id: string) => void;
  clearChat: () => void;
  inputText: string;
  setInputText: (t: string) => void;

  // Memory
  memories: Memory[];
  memoriesLoading: boolean;
  setMemories: (m: Memory[]) => void;
  setMemoriesLoading: (b: boolean) => void;
  removeMemory: (id: number) => void;

  // KG
  kgNodes: KGNode[];
  kgEdges: KGEdge[];
  kgLoading: boolean;
  setKG: (nodes: KGNode[], edges: KGEdge[]) => void;
  setKGLoading: (b: boolean) => void;

  // Projects
  activeProject: string | null;
  setActiveProject: (n: string | null) => void;

  // Skills
  skillStats: string;
  setSkillStats: (s: string) => void;

  // RAG docs
  ragDocs: RagDoc[];
  ragDocsLoading: boolean;
  setRagDocs: (d: RagDoc[]) => void;
  setRagDocsLoading: (b: boolean) => void;

  // System profile
  systemProfile: string;
  systemProfileLoading: boolean;
  setSystemProfile: (s: string) => void;
  setSystemProfileLoading: (b: boolean) => void;

  // Settings
  webSearchEnabled: boolean;
  setWebSearchEnabled: (b: boolean) => void;
  multiAgentEnabled: boolean;
  setMultiAgentEnabled: (b: boolean) => void;

  // Sidebar
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;
}

export const useStore = create<State>((set) => ({
  activePanel: "chat",
  setPanel: (p) => set({ activePanel: p }),

  health: null,
  setHealth: (h) => set({ health: h }),

  messages: [{ id: "welcome", role: "assistant", content: "Hafiz is ready.", ts: Date.now() }],
  isStreaming: false,
  addMessage: (m) => set((s) => ({ messages: [...s.messages, m] })),
  appendToken: (id, token) =>
    set((s) => ({
      isStreaming: true,
      messages: s.messages.map((m) => m.id === id ? { ...m, content: m.content + token } : m),
    })),
  finaliseStream: (id) =>
    set((s) => ({
      isStreaming: false,
      messages: s.messages.map((m) => m.id === id ? { ...m, streaming: false } : m),
    })),
  clearChat: () => set({ messages: [{ id: "welcome-2", role: "assistant", content: "Memory cleared.", ts: Date.now() }] }),
  inputText: "",
  setInputText: (t) => set({ inputText: t }),

  memories: [],
  memoriesLoading: false,
  setMemories: (m) => set({ memories: m }),
  setMemoriesLoading: (b) => set({ memoriesLoading: b }),
  removeMemory: (id) => set((s) => ({ memories: s.memories.filter((m) => m.id !== id) })),

  kgNodes: [],
  kgEdges: [],
  kgLoading: false,
  setKG: (nodes, edges) => set({ kgNodes: nodes, kgEdges: edges }),
  setKGLoading: (b) => set({ kgLoading: b }),

  activeProject: null,
  setActiveProject: (n) => set({ activeProject: n }),

  skillStats: "",
  setSkillStats: (s) => set({ skillStats: s }),

  ragDocs: [],
  ragDocsLoading: false,
  setRagDocs: (d) => set({ ragDocs: d }),
  setRagDocsLoading: (b) => set({ ragDocsLoading: b }),

  systemProfile: "",
  systemProfileLoading: false,
  setSystemProfile: (s) => set({ systemProfile: s }),
  setSystemProfileLoading: (b) => set({ systemProfileLoading: b }),

  webSearchEnabled: false,
  setWebSearchEnabled: (b) => set({ webSearchEnabled: b }),
  multiAgentEnabled: false,
  setMultiAgentEnabled: (b) => set({ multiAgentEnabled: b }),

  sidebarCollapsed: false,
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
}));
