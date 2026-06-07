/**
 * src/components/SystemPanel.tsx
 * System profiler + model recommender. Shows hardware + best model config.
 */
import React, { useState } from "react";
import { Cpu, RefreshCw, ChevronDown, ChevronRight } from "lucide-react";
import { useStore } from "../stores/useStore";
import { getSystemProfile } from "../api";

function Section({ title, content }: { title: string; content: string }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="profile-section">
      <button className="profile-section-header" onClick={() => setOpen(o => !o)}>
        {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        <span>{title}</span>
      </button>
      {open && <pre className="profile-pre">{content}</pre>}
    </div>
  );
}

function parseReport(raw: string): Record<string, string> {
  const sections: Record<string, string> = {};
  const sectionRe = /──\s+(.+?)\s+─+\n([\s\S]*?)(?=──|$)/g;
  let m;
  while ((m = sectionRe.exec(raw)) !== null) {
    sections[m[1].trim()] = m[2].trim();
  }
  return sections;
}

export function SystemPanel() {
  const { systemProfile, systemProfileLoading, setSystemProfile, setSystemProfileLoading } = useStore();
  const [error, setError] = useState("");

  const run = async () => {
    setSystemProfileLoading(true);
    setError("");
    try {
      const res = await getSystemProfile();
      setSystemProfile(res.result ?? "");
    } catch (e) {
      setError(String(e));
    } finally {
      setSystemProfileLoading(false);
    }
  };

  const sections = systemProfile ? parseReport(systemProfile) : {};
  const hasSections = Object.keys(sections).length > 0;

  // Highlight the env snippet section specially
  const envSnippet = sections["env snippet"] ?? sections[".env snippet"] ?? "";
  const recommendation = sections["Recommendation"] ?? "";

  return (
    <div className="panel-wrap">
      <div className="panel-header">
        <Cpu size={17} />
        <h2>System Profiler</h2>
        <div className="panel-header-actions">
          <button className="btn-icon" onClick={run} disabled={systemProfileLoading}>
            <RefreshCw size={13} className={systemProfileLoading ? "spin" : ""} />
          </button>
        </div>
      </div>

      <div className="system-panel-body">
        {!systemProfile && !systemProfileLoading && !error && (
          <div className="system-cta">
            <Cpu size={40} className="cta-icon" />
            <p className="cta-title">Analyse your hardware</p>
            <p className="cta-sub">
              Detects your RAM, GPU, CPU, and disk, then recommends the best
              LLM, quantisation, STT model, and GPU offload settings for this machine.
            </p>
            <button className="btn-run-profile" onClick={run}>
              Run System Profile
            </button>
          </div>
        )}

        {systemProfileLoading && (
          <div className="panel-loading">
            <RefreshCw size={18} className="spin" />
            <span style={{ marginLeft: 8 }}>Profiling hardware…</span>
          </div>
        )}

        {error && <div className="profile-error">{error}</div>}

        {hasSections && !systemProfileLoading && (
          <div className="profile-sections">
            {/* Recommendation first, highlighted */}
            {recommendation && (
              <div className="profile-recommendation">
                <div className="rec-header">Recommended Configuration</div>
                <pre className="rec-pre">{recommendation}</pre>
              </div>
            )}

            {/* env snippet, copiable */}
            {envSnippet && (
              <div className="profile-env-block">
                <div className="env-header">
                  <span>.env snippet — copy to your .env file</span>
                  <button className="btn-small" onClick={() => navigator.clipboard?.writeText(envSnippet)}>
                    Copy
                  </button>
                </div>
                <pre className="env-pre">{envSnippet}</pre>
              </div>
            )}

            {/* All other sections */}
            {Object.entries(sections)
              .filter(([k]) => k !== "Recommendation" && !k.toLowerCase().includes("env"))
              .map(([title, content]) => (
                <Section key={title} title={title} content={content} />
              ))
            }

            <button className="btn-small rerun-btn" onClick={run}>Re-run profile</button>
          </div>
        )}
      </div>
    </div>
  );
}
