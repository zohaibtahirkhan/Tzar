/**
 * src/components/Sidebar.tsx
 */
import React from "react";
import {
  MessageSquare, Brain, GitBranch, Layers,
  FolderOpen, Settings, ChevronLeft, ChevronRight,
  Circle, FileText, Cpu, FlaskConical
} from "lucide-react";
import { useStore, type Panel } from "../stores/useStore";

interface NavItem {
  id: Panel;
  label: string;
  icon: React.ReactNode;
}

const NAV: NavItem[] = [
  { id: "chat",     label: "Chat",      icon: <MessageSquare size={17} /> },
  { id: "memory",   label: "Memory",    icon: <Brain size={17} /> },
  { id: "graph",    label: "Graph",     icon: <GitBranch size={17} /> },
  { id: "docs",     label: "Documents", icon: <FileText size={17} /> },
  { id: "skills",   label: "Skills",    icon: <Layers size={17} /> },
  { id: "projects", label: "Projects",  icon: <FolderOpen size={17} /> },
  { id: "system",   label: "System",    icon: <Cpu size={17} /> },
  { id: "settings", label: "Settings",  icon: <Settings size={17} /> },
  { id: "tests", label: "Test Runner", icon: <FlaskConical size={17} /> },
];

export function Sidebar() {
  const { activePanel, setPanel, sidebarCollapsed, toggleSidebar, activeProject, health } = useStore();
  const online = health?.llm === "loaded";

  return (
    <aside className={`sidebar ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      {/* Logo */}
      <div className="sidebar-logo">
        {!sidebarCollapsed && (
          <div className="logo-text">
            <span className="logo-tzar">Tzar</span>
            {activeProject && <span className="logo-project">{activeProject}</span>}
          </div>
        )}
        <button className="btn-icon collapse-btn" onClick={toggleSidebar}>
          {sidebarCollapsed ? <ChevronRight size={13} /> : <ChevronLeft size={13} />}
        </button>
      </div>

      {/* Status */}
      <div className="sidebar-status">
        <Circle
          size={7}
          fill={online ? "var(--accent-green)" : "var(--accent-red)"}
          color={online ? "var(--accent-green)" : "var(--accent-red)"}
        />
        {!sidebarCollapsed && (
          <span className="status-text">{online ? "Online" : "Connecting…"}</span>
        )}
      </div>

      {/* Nav */}
      <nav className="sidebar-nav">
        {NAV.map((item) => (
          <button
            key={item.id}
            className={`nav-item ${activePanel === item.id ? "nav-active" : ""}`}
            onClick={() => setPanel(item.id)}
            title={sidebarCollapsed ? item.label : undefined}
          >
            <span className="nav-icon">{item.icon}</span>
            {!sidebarCollapsed && <span className="nav-label">{item.label}</span>}
          </button>
        ))}
      </nav>
    </aside>
  );
}