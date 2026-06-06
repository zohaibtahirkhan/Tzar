/**
 * src/stores/useStore.ts
 *
 * Single Zustand store for the entire app.
 * Panels: chat, memory, knowledge-graph, skills, projects, settings.
 */

import { create } from "zustand";
import type { Memory, HealthResponse } from "../api";

// ─── Types ────────────────────────────────────────────────────────────────────

export type Panel = "chat" | "memory" | "graph" | "skills" | "projects" | "settings";

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  streaming?: boolean;
  toolResults?: { tool: string; result: string }[];
  ts: number;
}

export interface Project {
  name: string;
  status: "active" | "paused" | "archived";
  description: string;
  last_opened: string | null;
}

export interface KGNode {
  id: string;
  label: string;
  group: string;
}

export interface KGEdge {
  source: string;
  target: string;
  relation: string;
}

// ─── Store ────────────────────────────────────────────────────────────────────

interface State {
  // Navigation
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

  // Input
  inputText: string;
  setInputText: (t: string) => void;

  // Memories
  memories: Memory[];
  memoriesLoading: boolean;
  setMemories: (m: Memory[]) => void;
  setMemoriesLoading: (b: boolean) => void;
  removeMemory: (id: number) => void;

  // Knowledge Graph
  kgNodes: KGNode[];
  kgEdges: KGEdge[];
  kgLoading: boolean;
  setKG: (nodes: KGNode[], edges: KGEdge[]) => void;
  setKGLoading: (b: boolean) => void;

  // Projects
  projects: Project[];
  activeProject: string | null;
  projectsLoading: boolean;
  setProjects: (p: Project[]) => void;
  setActiveProject: (name: string | null) => void;
  setProjectsLoading: (b: boolean) => void;

  // Skills
  skillStats: string;
  setSkillStats: (s: string) => void;

  // Settings
  webSearchEnabled: boolean;
  setWebSearchEnabled: (b: boolean) => void;

  // Sidebar
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;
}

let _msgId = 0;
export const nextId = () => `msg-${++_msgId}-${Date.now()}`;

export const useStore = create<State>((set) => ({
  // Navigation
  activePanel: "chat",
  setPanel: (p) => set({ activePanel: p }),

  // Health
  health: null,
  setHealth: (h) => set({ health: h }),

  // Chat
  messages: [
    {
      id: "welcome",
      role: "assistant",
      content: "Hafiz is ready. How can I help?",
      ts: Date.now(),
    },
  ],
  isStreaming: false,
  addMessage: (m) =>
    set((s) => ({ messages: [...s.messages, m] })),
  appendToken: (id, token) =>
    set((s) => ({
      isStreaming: true,
      messages: s.messages.map((m) =>
        m.id === id ? { ...m, content: m.content + token } : m
      ),
    })),
  finaliseStream: (id) =>
    set((s) => ({
      isStreaming: false,
      messages: s.messages.map((m) =>
        m.id === id ? { ...m, streaming: false } : m
      ),
    })),
  clearChat: () =>
    set({
      messages: [
        {
          id: "welcome",
          role: "assistant",
          content: "Memory cleared. Fresh start.",
          ts: Date.now(),
        },
      ],
    }),

  // Input
  inputText: "",
  setInputText: (t) => set({ inputText: t }),

  // Memories
  memories: [],
  memoriesLoading: false,
  setMemories: (m) => set({ memories: m }),
  setMemoriesLoading: (b) => set({ memoriesLoading: b }),
  removeMemory: (id) =>
    set((s) => ({ memories: s.memories.filter((m) => m.id !== id) })),

  // KG
  kgNodes: [],
  kgEdges: [],
  kgLoading: false,
  setKG: (nodes, edges) => set({ kgNodes: nodes, kgEdges: edges }),
  setKGLoading: (b) => set({ kgLoading: b }),

  // Projects
  projects: [],
  activeProject: null,
  projectsLoading: false,
  setProjects: (p) => set({ projects: p }),
  setActiveProject: (name) => set({ activeProject: name }),
  setProjectsLoading: (b) => set({ projectsLoading: b }),

  // Skills
  skillStats: "",
  setSkillStats: (s) => set({ skillStats: s }),

  // Settings
  webSearchEnabled: false,
  setWebSearchEnabled: (b) => set({ webSearchEnabled: b }),

  // Sidebar
  sidebarCollapsed: false,
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
}));
