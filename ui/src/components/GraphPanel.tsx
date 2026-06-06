/**
 * src/components/GraphPanel.tsx
 *
 * D3 force-directed knowledge graph visualisation.
 * Nodes: entities/concepts. Edges: typed relations.
 * Click a node to expand its neighbours (calls kg_neighbors).
 */

import React, { useEffect, useRef, useCallback, useState } from "react";
import * as d3 from "d3";
import { GitBranch, RefreshCw, ZoomIn, ZoomOut, Maximize2 } from "lucide-react";
import { useStore, type KGNode, type KGEdge } from "../stores/useStore";
import { invokeTool } from "../api";

// ─── D3 types ─────────────────────────────────────────────────────────────────

interface SimNode extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  group: string;
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  relation: string;
}

// ─── Colour by group ──────────────────────────────────────────────────────────

const GROUP_COLORS: Record<string, string> = {
  concept:  "#7c3aed",
  note:     "#06b6d4",
  person:   "#f59e0b",
  project:  "#10b981",
  tool:     "#f43f5e",
  default:  "#64748b",
};

function groupColor(group: string) {
  return GROUP_COLORS[group] ?? GROUP_COLORS.default;
}

// ─── Parse KG summary text → nodes + edges ────────────────────────────────────

function parseKGSummary(raw: string): { nodes: KGNode[]; edges: KGEdge[] } {
  // The kg_summary tool returns plain text stats — we use kg_path / kg_neighbors
  // for structured data. For the graph, we hit kg_neighbors with "*" or parse
  // a full dump. For now, parse the raw text as best we can.
  const nodes: KGNode[] = [];
  const edges: KGEdge[] = [];
  const seen = new Set<string>();

  // Look for "→" edge lines: "  related_to → TargetName (label)"
  const edgeRe = /(\w[\w\s]+)\s+—\[(.+?)\]→\s+(\w[\w\s]+)/g;
  let m;
  while ((m = edgeRe.exec(raw)) !== null) {
    const [, src, rel, tgt] = m;
    const s = src.trim();
    const t = tgt.trim();
    if (!seen.has(s)) { seen.add(s); nodes.push({ id: s, label: s, group: "concept" }); }
    if (!seen.has(t)) { seen.add(t); nodes.push({ id: t, label: t, group: "concept" }); }
    edges.push({ source: s, target: t, relation: rel.trim() });
  }

  return { nodes, edges };
}

// ─── Graph Panel ──────────────────────────────────────────────────────────────

export function GraphPanel() {
  const svgRef = useRef<SVGSVGElement>(null);
  const { kgNodes, kgEdges, kgLoading, setKG, setKGLoading } = useStore();
  const [selected, setSelected] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);
  const simRef = useRef<d3.Simulation<SimNode, SimLink> | null>(null);

  // ── Load data ───────────────────────────────────────────────────────────────

  const loadGraph = useCallback(async () => {
    setKGLoading(true);
    try {
      // Get full graph dump via summary + top clusters
      const summaryRes = await invokeTool("kg_summary");
      const { nodes, edges } = parseKGSummary(summaryRes.result ?? "");

      // Also try to get cluster data to populate nodes
      const clusterRes = await invokeTool("kg_clusters", { min_size: 2 });
      const clusterText = clusterRes.result ?? "";
      const clusterRe = /Cluster\s+\d+[^:]*:\s+(.+)/g;
      let cm;
      const allNames = new Set(nodes.map((n) => n.id));
      while ((cm = clusterRe.exec(clusterText)) !== null) {
        cm[1].split(",").forEach((name) => {
          const n = name.trim().replace(/\s*\(\+\d+ more\)/, "");
          if (n && !allNames.has(n)) {
            allNames.add(n);
            nodes.push({ id: n, label: n, group: "concept" });
          }
        });
      }

      setKG(nodes, edges);
    } catch (err) {
      console.error("KG load failed:", err);
    } finally {
      setKGLoading(false);
    }
  }, [setKG, setKGLoading]);

  const expandNode = useCallback(
    async (nodeId: string) => {
      try {
        const res = await invokeTool("kg_neighbors", { node: nodeId, depth: 1 });
        const { nodes: newNodes, edges: newEdges } = parseKGSummary(res.result ?? "");
        const existingIds = new Set(kgNodes.map((n) => n.id));
        const existingSrcs = new Set(
          kgEdges.map((e) => `${e.source}|${e.target}|${e.relation}`)
        );
        const addNodes = newNodes.filter((n) => !existingIds.has(n.id));
        const addEdges = newEdges.filter(
          (e) => !existingSrcs.has(`${e.source}|${e.target}|${e.relation}`)
        );
        setKG([...kgNodes, ...addNodes], [...kgEdges, ...addEdges]);
      } catch {
        /* ignore */
      }
    },
    [kgNodes, kgEdges, setKG]
  );

  useEffect(() => {
    loadGraph();
  }, []);

  // ── D3 render ───────────────────────────────────────────────────────────────

  useEffect(() => {
    const svg = d3.select(svgRef.current!);
    svg.selectAll("*").remove();

    if (kgNodes.length === 0) return;

    const rect = svgRef.current!.getBoundingClientRect();
    const W = rect.width || 800;
    const H = rect.height || 500;

    const simNodes: SimNode[] = kgNodes.map((n) => ({ ...n }));
    const idToNode = new Map(simNodes.map((n) => [n.id, n]));

    const simLinks: SimLink[] = kgEdges
      .map((e) => ({
        source: idToNode.get(e.source) ?? e.source,
        target: idToNode.get(e.target) ?? e.target,
        relation: e.relation,
      }))
      .filter((l) => typeof l.source === "object" && typeof l.target === "object");

    // Zoom container
    const g = svg.append("g");
    const zoom = d3.zoom<SVGSVGElement, unknown>().scaleExtent([0.2, 4]).on("zoom", (event) => {
      g.attr("transform", event.transform);
    });
    svg.call(zoom);

    // Simulation
    const sim = d3
      .forceSimulation<SimNode>(simNodes)
      .force("link", d3.forceLink<SimNode, SimLink>(simLinks).id((d) => d.id).distance(90))
      .force("charge", d3.forceManyBody().strength(-220))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force("collision", d3.forceCollide(24));
    simRef.current = sim;

    // Edges
    const linkSel = g
      .append("g")
      .selectAll("line")
      .data(simLinks)
      .enter()
      .append("line")
      .attr("stroke", "rgba(255,255,255,0.1)")
      .attr("stroke-width", 1.2);

    // Edge labels
    const linkLabelSel = g
      .append("g")
      .selectAll("text")
      .data(simLinks)
      .enter()
      .append("text")
      .text((d) => d.relation)
      .attr("font-size", 8)
      .attr("fill", "rgba(255,255,255,0.3)")
      .attr("text-anchor", "middle");

    // Nodes
    const nodeSel = g
      .append("g")
      .selectAll("g")
      .data(simNodes)
      .enter()
      .append("g")
      .attr("cursor", "pointer")
      .call(
        d3
          .drag<SVGGElement, SimNode>()
          .on("start", (event, d) => {
            if (!event.active) sim.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) sim.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    nodeSel
      .append("circle")
      .attr("r", 14)
      .attr("fill", (d) => groupColor(d.group))
      .attr("fill-opacity", 0.85)
      .attr("stroke", "rgba(255,255,255,0.15)")
      .attr("stroke-width", 1);

    nodeSel
      .append("text")
      .text((d) => (d.label.length > 10 ? d.label.slice(0, 9) + "…" : d.label))
      .attr("font-size", 9)
      .attr("fill", "#e2e8f0")
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em");

    nodeSel
      .on("click", (_event, d) => {
        setSelected(d.id);
        expandNode(d.id);
      })
      .on("mouseenter", (event, d) => {
        const [mx, my] = d3.pointer(event, svgRef.current!);
        setTooltip({ x: mx, y: my - 20, text: `${d.label} (${d.group})` });
      })
      .on("mouseleave", () => setTooltip(null));

    sim.on("tick", () => {
      linkSel
        .attr("x1", (d) => (d.source as SimNode).x ?? 0)
        .attr("y1", (d) => (d.source as SimNode).y ?? 0)
        .attr("x2", (d) => (d.target as SimNode).x ?? 0)
        .attr("y2", (d) => (d.target as SimNode).y ?? 0);

      linkLabelSel
        .attr("x", (d) => (((d.source as SimNode).x ?? 0) + ((d.target as SimNode).x ?? 0)) / 2)
        .attr("y", (d) => (((d.source as SimNode).y ?? 0) + ((d.target as SimNode).y ?? 0)) / 2);

      nodeSel.attr("transform", (d) => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });

    return () => {
      sim.stop();
    };
  }, [kgNodes, kgEdges, expandNode]);

  const zoomIn  = () => d3.select(svgRef.current!).transition().call(d3.zoom<SVGSVGElement, unknown>().scaleBy as never, 1.4);
  const zoomOut = () => d3.select(svgRef.current!).transition().call(d3.zoom<SVGSVGElement, unknown>().scaleBy as never, 0.7);

  return (
    <div className="panel-wrap">
      <div className="panel-header">
        <GitBranch size={18} />
        <h2>Knowledge Graph</h2>
        <div className="panel-header-actions">
          <span className="stat-chip">{kgNodes.length} nodes</span>
          <span className="stat-chip">{kgEdges.length} edges</span>
          <button className="btn-icon" onClick={zoomIn}><ZoomIn size={14} /></button>
          <button className="btn-icon" onClick={zoomOut}><ZoomOut size={14} /></button>
          <button className="btn-icon" onClick={loadGraph}><RefreshCw size={14} /></button>
        </div>
      </div>

      <div className="kg-legend">
        {Object.entries(GROUP_COLORS).filter(([k]) => k !== "default").map(([group, color]) => (
          <div key={group} className="legend-item">
            <span className="legend-dot" style={{ background: color }} />
            <span>{group}</span>
          </div>
        ))}
      </div>

      <div className="kg-container">
        {kgLoading && <div className="kg-loading">Loading graph…</div>}
        {!kgLoading && kgNodes.length === 0 && (
          <div className="panel-empty">
            No knowledge graph data yet.<br />
            Create notes to start building your graph.
          </div>
        )}
        <svg ref={svgRef} className="kg-svg" />

        {tooltip && (
          <div
            className="kg-tooltip"
            style={{ left: tooltip.x, top: tooltip.y }}
          >
            {tooltip.text}
          </div>
        )}
      </div>

      {selected && (
        <div className="kg-selected">
          Selected: <strong>{selected}</strong>
          <button className="btn-small" onClick={() => setSelected(null)}>×</button>
        </div>
      )}
    </div>
  );
}
