/**
 * src/App.tsx
 */

import React, { useEffect } from "react";
import { Sidebar } from "./components/Sidebar";
import { ChatPanel } from "./components/ChatPanel";
import { MemoryPanel } from "./components/MemoryPanel";
import { GraphPanel } from "./components/GraphPanel";
import {
  SkillsPanel,
  ProjectsPanel,
  SettingsPanel,
} from "./components/SecondaryPanels";
import { useStore } from "./stores/useStore";
import { getHealth } from "./api";

export default function App() {
  const { activePanel, setHealth, setWebSearchEnabled } = useStore();

  // Poll health every 10s
  useEffect(() => {
    const poll = async () => {
      try {
        const h = await getHealth();
        setHealth(h);
        setWebSearchEnabled(h.web_search);
      } catch {
        /* backend not ready yet */
      }
    };
    poll();
    const id = setInterval(poll, 10_000);
    return () => clearInterval(id);
  }, [setHealth, setWebSearchEnabled]);

  return (
    <div className="app-shell">
      <Sidebar />
      <main className="main-content">
        {activePanel === "chat"     && <ChatPanel />}
        {activePanel === "memory"   && <MemoryPanel />}
        {activePanel === "graph"    && <GraphPanel />}
        {activePanel === "skills"   && <SkillsPanel />}
        {activePanel === "projects" && <ProjectsPanel />}
        {activePanel === "settings" && <SettingsPanel />}
      </main>
    </div>
  );
}
