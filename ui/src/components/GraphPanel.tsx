/**
 * src/components/GraphPanel.tsx
 * D3 force-directed KG. Click node to expand. Node search. Path finder.
 */
import React, { useEffect, useRef, useCallback, useState } from "react";
import * as d3 from "d3";
import { GitBranch, RefreshCw, ZoomIn, ZoomOut, Search } from "lucide-react";
import { useStore, type KGNode, type KGEdge } from "../stores/useStore";
import { invokeTool } from "../api";

interface SimNode extends d3.SimulationNodeDatum { id: string; label: string; group: string; }
interface SimLink extends d3.SimulationLinkDatum<SimNode> { relation: string; }

const COLORS: Record<string, string> = {
  concept: "#7c3aed", note: "#06b6d4", person: "#f59e0b",
  project: "#10b981", tool: "#f43f5e", default: "#64748b",
};
const nodeColor = (g: string) => COLORS[g] ?? COLORS.default;

// ─── KG text parser — preserves node groups from backend output ───────────────

function parseKGText(raw: string): { nodes: KGNode[]; edges: KGEdge[] } {
  const nodes: KGNode[] = [];
  const edges: KGEdge[] = [];
  const seen = new Set<string>();

  // Match: "NodeA [group]? —[relation]→ NodeB [group]?"
  // Group is optional — defaults to "concept"
  const edgeRe = /([\w][\w\s]+?)\s*(?:\[(\w+)\])?\s*—\[(.+?)\]→\s*([\w][\w\s]+?)\s*(?:\[(\w+)\])?(?:\s|$)/g;
  let m;
  while ((m = edgeRe.exec(raw)) !== null) {
    const src     = m[1].trim();
    const srcGrp  = m[2]?.trim() ?? "concept";
    const rel     = m[3].trim();
    const tgt     = m[4].trim();
    const tgtGrp  = m[5]?.trim() ?? "concept";

    if (!seen.has(src)) { seen.add(src); nodes.push({ id: src, label: src, group: srcGrp }); }
    if (!seen.has(tgt)) { seen.add(tgt); nodes.push({ id: tgt, label: tgt, group: tgtGrp }); }
    edges.push({ source: src, target: tgt, relation: rel });
  }

  // Cluster lines: "Cluster N (Type): NodeA, NodeB, ..."
  const clusterRe = /Cluster\s+\d+\s*(?:\((\w+)\))?[^:]*:\s+(.+)/g;
  while ((m = clusterRe.exec(raw)) !== null) {
    const grp = m[1]?.toLowerCase() ?? "concept";
    m[2].split(",").forEach(name => {
      const n = name.trim().replace(/\s*\(\+\d+ more\)/, "");
      if (n && !seen.has(n)) { seen.add(n); nodes.push({ id: n, label: n, group: grp }); }
    });
  }

  return { nodes, edges };
}

export function GraphPanel() {
  const svgRef  = useRef<SVGSVGElement>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown>>();
  const { kgNodes, kgEdges, kgLoading, setKG, setKGLoading } = useStore();
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);
  const [pathA, setPathA]     = useState("");
  const [pathB, setPathB]     = useState("");
  const [pathResult, setPathResult] = useState("");

  const loadGraph = useCallback(async () => {
    setKGLoading(true);
    try {
      const [summaryRes, clusterRes] = await Promise.all([
        invokeTool("kg_summary"),
        invokeTool("kg_clusters", { min_size: 2 }),
      ]);
      const combined = (summaryRes.result ?? "") + "\n" + (clusterRes.result ?? "");
      const { nodes, edges } = parseKGText(combined);
      setKG(nodes, edges);
    } catch (e) { console.error(e); } finally { setKGLoading(false); }
  }, [setKG, setKGLoading]);

  const expandNode = useCallback(async (nodeId: string) => {
    const res = await invokeTool("kg_neighbors", { node: nodeId, depth: 1 });
    const { nodes: newN, edges: newE } = parseKGText(res.result ?? "");
    const existIds   = new Set(kgNodes.map(n => n.id));
    const existEdges = new Set(kgEdges.map(e => `${e.source}|${e.target}|${e.relation}`));
    setKG(
      [...kgNodes, ...newN.filter(n => !existIds.has(n.id))],
      [...kgEdges, ...newE.filter(e => !existEdges.has(`${e.source}|${e.target}|${e.relation}`))]
    );
  }, [kgNodes, kgEdges, setKG]);

  useEffect(() => { loadGraph(); }, []);

  // D3 render
  useEffect(() => {
    const svg = d3.select(svgRef.current!);
    svg.selectAll("*").remove();
    if (kgNodes.length === 0) return;

    const rect = svgRef.current!.getBoundingClientRect();
    const W = rect.width || 800, H = rect.height || 500;
    const simNodes: SimNode[] = kgNodes.map(n => ({ ...n }));
    const idMap = new Map(simNodes.map(n => [n.id, n]));
    const simLinks: SimLink[] = kgEdges
      .map(e => ({ source: idMap.get(e.source as string) ?? e.source, target: idMap.get(e.target as string) ?? e.target, relation: e.relation }))
      .filter(l => typeof l.source === "object" && typeof l.target === "object");

    const g    = svg.append("g");
    const zoom = d3.zoom<SVGSVGElement, unknown>().scaleExtent([0.15, 5]).on("zoom", ev => g.attr("transform", ev.transform));
    zoomRef.current = zoom;
    svg.call(zoom);

    const sim = d3.forceSimulation<SimNode>(simNodes)
      .force("link",    d3.forceLink<SimNode, SimLink>(simLinks).id(d => d.id).distance(85))
      .force("charge",  d3.forceManyBody().strength(-200))
      .force("center",  d3.forceCenter(W / 2, H / 2))
      .force("collide", d3.forceCollide(22));

    const linkSel = g.append("g").selectAll("line").data(simLinks).enter().append("line")
      .attr("stroke", "rgba(255,255,255,0.08)").attr("stroke-width", 1.2);

    const linkLabelSel = g.append("g").selectAll("text").data(simLinks).enter().append("text")
      .text(d => d.relation).attr("font-size", 7.5).attr("fill", "rgba(255,255,255,0.25)").attr("text-anchor", "middle");

    const nodeSel = g.append("g").selectAll("g").data(simNodes).enter().append("g").attr("cursor", "pointer")
      .call(d3.drag<SVGGElement, SimNode>()
        .on("start", (ev, d) => { if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag",  (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
        .on("end",   (ev, d) => { if (!ev.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    nodeSel.append("circle").attr("r", 13)
      .attr("fill", d => nodeColor(d.group)).attr("fill-opacity", 0.85)
      .attr("stroke", "rgba(255,255,255,0.12)").attr("stroke-width", 1);

    nodeSel.append("text")
      .text(d => d.label.length > 9 ? d.label.slice(0, 8) + "…" : d.label)
      .attr("font-size", 8.5).attr("fill", "#e2e8f0").attr("text-anchor", "middle").attr("dy", "0.35em");

    nodeSel
      .on("click", (_ev, d) => expandNode(d.id))
      .on("mouseenter", (ev, d) => {
        const [x, y] = d3.pointer(ev, svgRef.current!);
        setTooltip({ x, y: y - 18, text: `${d.label} · ${d.group}` });
      })
      .on("mouseleave", () => setTooltip(null));

    sim.on("tick", () => {
      linkSel
        .attr("x1", d => (d.source as SimNode).x ?? 0).attr("y1", d => (d.source as SimNode).y ?? 0)
        .attr("x2", d => (d.target as SimNode).x ?? 0).attr("y2", d => (d.target as SimNode).y ?? 0);
      linkLabelSel
        .attr("x", d => (((d.source as SimNode).x ?? 0) + ((d.target as SimNode).x ?? 0)) / 2)
        .attr("y", d => (((d.source as SimNode).y ?? 0) + ((d.target as SimNode).y ?? 0)) / 2);
      nodeSel.attr("transform", d => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });

    return () => { sim.stop(); };
  }, [kgNodes, kgEdges, expandNode]);

  const doZoom = (factor: number) => {
    if (zoomRef.current) d3.select(svgRef.current!).transition().duration(250).call(zoomRef.current.scaleBy, factor);
  };

  const findPath = async () => {
    if (!pathA.trim() || !pathB.trim()) return;
    const res = await invokeTool("kg_path", { source: pathA.trim(), target: pathB.trim() });
    setPathResult(res.result ?? "No path found.");
  };

  return (
    <div className="panel-wrap">
      <div className="panel-header">
        <GitBranch size={17} />
        <h2>Knowledge Graph</h2>
        <div className="panel-header-actions">
          <span className="stat-chip">{kgNodes.length} nodes</span>
          <span className="stat-chip">{kgEdges.length} edges</span>
          <button className="btn-icon" onClick={() => doZoom(1.4)}><ZoomIn size={13} /></button>
          <button className="btn-icon" onClick={() => doZoom(0.7)}><ZoomOut size={13} /></button>
          <button className="btn-icon" onClick={loadGraph}><RefreshCw size={13} /></button>
        </div>
      </div>

      {/* Legend */}
      <div className="kg-legend">
        {Object.entries(COLORS).filter(([k]) => k !== "default").map(([g, c]) => (
          <div key={g} className="legend-item">
            <span className="legend-dot" style={{ background: c }} />
            <span>{g}</span>
          </div>
        ))}
      </div>

      {/* Path finder */}
      <div className="kg-path-bar">
        <input className="path-input" value={pathA} onChange={e => setPathA(e.target.value)} placeholder="From node…" />
        <span className="path-arrow">→</span>
        <input className="path-input" value={pathB} onChange={e => setPathB(e.target.value)} placeholder="To node…" />
        <button className="btn-small" onClick={findPath}><Search size={11} /> Path</button>
      </div>
      {pathResult && <div className="path-result">{pathResult}</div>}

      {/* Graph */}
      <div className="kg-container">
        {kgLoading && <div className="kg-loading">Loading graph…</div>}
        {!kgLoading && kgNodes.length === 0 && (
          <div className="panel-empty">No graph data yet.<br />Create notes to start building your graph.</div>
        )}
        <svg ref={svgRef} className="kg-svg" />
        {tooltip && (
          <div className="kg-tooltip" style={{ left: tooltip.x, top: tooltip.y }}>{tooltip.text}</div>
        )}
      </div>
    </div>
  );
}
