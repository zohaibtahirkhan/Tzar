/**
 * src/App.tsx
 */
import React, { useEffect } from "react";
import { Sidebar } from "./components/Sidebar";
import { ChatPanel } from "./components/ChatPanel";
import { MemoryPanel } from "./components/MemoryPanel";
import { GraphPanel } from "./components/GraphPanel";
import { DocsPanel } from "./components/DocsPanel";
import { SystemPanel } from "./components/SystemPanel";
import { SkillsPanel, ProjectsPanel, SettingsPanel } from "./components/SecondaryPanels";
import { TestPanel } from "./components/TestPanel";
import { useStore } from "./stores/useStore";
import { getHealth } from "./api";

export default function App() {
  const { activePanel, setHealth, setWebSearchEnabled } = useStore();

  useEffect(() => {
    const poll = async () => {
      try {
        const h = await getHealth();
        setHealth(h);
        setWebSearchEnabled(h.web_search);
      } catch { /* backend not ready */ }
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
        {activePanel === "docs"     && <DocsPanel />}
        {activePanel === "skills"   && <SkillsPanel />}
        {activePanel === "projects" && <ProjectsPanel />}
        {activePanel === "system"   && <SystemPanel />}
        {activePanel === "settings" && <SettingsPanel />}
        {activePanel === "tests" && <TestPanel />}
      </main>
    </div>
  );
}
