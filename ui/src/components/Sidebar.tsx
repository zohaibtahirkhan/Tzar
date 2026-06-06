/**
 * src/components/Sidebar.tsx
 */

import React from "react";
import {
  MessageSquare, Brain, GitBranch, Layers,
  FolderOpen, Settings, ChevronLeft, ChevronRight,
  Circle,
} from "lucide-react";
import { useStore, type Panel } from "../stores/useStore";

interface NavItem {
  id: Panel;
  label: string;
  icon: React.ReactNode;
}

const NAV: NavItem[] = [
  { id: "chat",     label: "Chat",     icon: <MessageSquare size={18} /> },
  { id: "memory",   label: "Memory",   icon: <Brain size={18} /> },
  { id: "graph",    label: "Graph",    icon: <GitBranch size={18} /> },
  { id: "skills",   label: "Skills",   icon: <Layers size={18} /> },
  { id: "projects", label: "Projects", icon: <FolderOpen size={18} /> },
  { id: "settings", label: "Settings", icon: <Settings size={18} /> },
];

export function Sidebar() {
  const { activePanel, setPanel, sidebarCollapsed, toggleSidebar, activeProject, health } =
    useStore();

  return (
    <aside className={`sidebar ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      {/* Logo */}
      <div className="sidebar-logo">
        {!sidebarCollapsed && (
          <div className="logo-text">
            <span className="logo-hafiz">Hafiz</span>
            {activeProject && (
              <span className="logo-project">{activeProject}</span>
            )}
          </div>
        )}
        <button className="btn-icon collapse-btn" onClick={toggleSidebar}>
          {sidebarCollapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
        </button>
      </div>

      {/* Status dot */}
      <div className="sidebar-status">
        <Circle
          size={7}
          fill={health?.llm === "loaded" ? "var(--accent-green)" : "var(--accent-red)"}
          color={health?.llm === "loaded" ? "var(--accent-green)" : "var(--accent-red)"}
        />
        {!sidebarCollapsed && (
          <span className="status-text">
            {health?.llm === "loaded" ? "Online" : "Starting…"}
          </span>
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
            {item.icon}
            {!sidebarCollapsed && <span className="nav-label">{item.label}</span>}
          </button>
        ))}
      </nav>

      {/* Footer */}
      {!sidebarCollapsed && (
        <div className="sidebar-footer">
          <span className="footer-text">حافظ</span>
        </div>
      )}
    </aside>
  );
}
