/**
 * src/components/GraphPanel.tsx
 * D3 force-directed KG. Click node to expand. Node search. Path finder. Timeline view.
 */
import React, { useEffect, useRef, useCallback, useState } from "react";
import * as d3 from "d3";
import { GitBranch, RefreshCw, ZoomIn, ZoomOut, Search, Clock, Network } from "lucide-react";
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

interface NodeDetail {
  node: KGNode;
  relations: Array<{ target: string; relation: string; direction: "out" }
                  | { source: string; relation: string; direction: "in" }>;
}

export function GraphPanel() {
  const svgRef  = useRef<SVGSVGElement>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown>>();
  const { kgNodes, kgEdges, kgLoading, setKG, setKGLoading } = useStore();
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);
  const [pathA, setPathA]     = useState("");
  const [pathB, setPathB]     = useState("");
  const [pathResult, setPathResult] = useState("");
  const [viewMode, setViewMode] = useState<"graph" | "timeline">("graph");
  const [timelineData, setTimelineData] = useState<string>("");
  const [selectedNode, setSelectedNode] = useState<NodeDetail | null>(null);
  const [showOrphans, setShowOrphans] = useState(true);
  const [orphanNodes, setOrphanNodes] = useState<Set<string>>(new Set());

  const loadGraph = useCallback(async () => {
    setKGLoading(true);
    try {
      const [summaryRes, clusterRes, orphansRes] = await Promise.all([
        invokeTool("kg_summary"),
        invokeTool("kg_clusters", { min_size: 2 }),
        invokeTool("kg_orphans"),
      ]);
      const combined = (summaryRes.result ?? "") + "\n" + (clusterRes.result ?? "");
      const { nodes, edges } = parseKGText(combined);
      
      // Parse orphan nodes and add them to nodes array
      const orphanText = orphansRes.result ?? "";
      const orphanMatches = orphanText.match(/^(?!Orphan nodes)(\w[\w\s]+?)\s*\[(\w+)\]/gm);
      const orphans = new Set<string>();
      
      if (orphanMatches) {
        orphanMatches.forEach(match => {
          const parts = match.match(/^(\w[\w\s]+?)\s*\[(\w+)\]/);
          if (parts) {
            const name = parts[1].trim();
            const label = parts[2].trim();
            orphans.add(name);
            
            // Add orphan node if not already in nodes array
            if (!nodes.find(n => n.id === name)) {
              nodes.push({ id: name, label: name, group: label });
            }
          }
        });
      }
      
      setOrphanNodes(orphans);
      setKG(nodes, edges);
    } catch (e) { console.error(e); } finally { setKGLoading(false); }
  }, [setKG, setKGLoading]);

  const loadTimeline = useCallback(async () => {
    setKGLoading(true);
    try {
      // Try to get orphan nodes which show recent additions
      const orphansRes = await invokeTool("kg_orphans");
      setTimelineData(orphansRes.result || "No timeline data available. Try adding nodes to the knowledge graph first.");
    } catch (e) {
      setTimelineData("Unable to load timeline data.");
    } finally {
      setKGLoading(false);
    }
  }, [setKGLoading]);

  const getNodeDetails = useCallback((nodeId: string): NodeDetail | null => {
    const node = kgNodes.find(n => n.id === nodeId);
    if (!node) return null;

    const relations: NodeDetail["relations"] = [];
    
    // Outgoing edges
    kgEdges.forEach(edge => {
      if (edge.source === nodeId || (typeof edge.source === 'object' && (edge.source as any).id === nodeId)) {
        const targetId = typeof edge.target === 'string' ? edge.target : (edge.target as any).id;
        relations.push({ target: targetId, relation: edge.relation, direction: "out" });
      }
    });
    
    // Incoming edges
    kgEdges.forEach(edge => {
      if (edge.target === nodeId || (typeof edge.target === 'object' && (edge.target as any).id === nodeId)) {
        const sourceId = typeof edge.source === 'string' ? edge.source : (edge.source as any).id;
        relations.push({ source: sourceId, relation: edge.relation, direction: "in" });
      }
    });

    return { node, relations };
  }, [kgNodes, kgEdges]);

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

  useEffect(() => { 
    if (viewMode === "graph") {
      loadGraph();
    } else {
      loadTimeline();
    }
  }, [viewMode, loadGraph, loadTimeline]);

  // D3 render
  useEffect(() => {
    const svg = d3.select(svgRef.current!);
    svg.selectAll("*").remove();
    if (kgNodes.length === 0) return;

    const rect = svgRef.current!.getBoundingClientRect();
    const W = rect.width || 800, H = rect.height || 500;
    
    // Filter nodes based on orphan toggle
    const filteredNodes = showOrphans 
      ? kgNodes 
      : kgNodes.filter(n => !orphanNodes.has(n.id));
    
    const simNodes: SimNode[] = filteredNodes.map(n => ({ ...n }));
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
      .attr("stroke", "rgba(124, 58, 237, 0.35)")  // Purple with better visibility
      .attr("stroke-width", 2)
      .attr("marker-end", "url(#arrowhead)");  // Add arrow marker
    
    // Add arrowhead marker definition
    svg.append("defs").append("marker")
      .attr("id", "arrowhead")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 20)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "rgba(124, 58, 237, 0.5)");

    const linkLabelSel = g.append("g").selectAll("text").data(simLinks).enter().append("text")
      .text(d => d.relation)
      .attr("font-size", 9)
      .attr("fill", "rgba(255,255,255,0.6)")  // More visible
      .attr("text-anchor", "middle")
      .attr("font-weight", 600)
      .attr("pointer-events", "none")
      .style("text-shadow", "0 0 3px rgba(0,0,0,0.8)");  // Add text shadow for readability

    const nodeSel = g.append("g").selectAll("g").data(simNodes).enter().append("g").attr("cursor", "pointer")
      .call(d3.drag<SVGGElement, SimNode>()
        .on("start", (ev, d) => { if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag",  (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
        .on("end",   (ev, d) => { if (!ev.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    nodeSel.append("circle").attr("r", 13)
      .attr("fill", d => nodeColor(d.group))
      .attr("fill-opacity", 0.95)  // More opaque
      .attr("stroke", "rgba(255,255,255,0.3)")  // More visible border
      .attr("stroke-width", 2);

    nodeSel.append("text")
      .text(d => d.label.length > 12 ? d.label.slice(0, 11) + "…" : d.label)
      .attr("font-size", 9)
      .attr("fill", "#fff")
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em")
      .attr("font-weight", 600)
      .attr("pointer-events", "none")
      .style("text-shadow", "0 1px 3px rgba(0,0,0,0.8)");  // Better text visibility

    nodeSel
      .on("click", (_ev, d) => {
        const details = getNodeDetails(d.id);
        if (details) {
          setSelectedNode(details);
        }
      })
      .on("mouseenter", (ev, d) => {
        const [x, y] = d3.pointer(ev, svgRef.current!);
        setTooltip({ x, y: y - 18, text: `${d.label} [${d.group}]` });
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
  }, [kgNodes, kgEdges, expandNode, getNodeDetails, showOrphans, orphanNodes]);

  const doZoom = (factor: number) => {
    if (zoomRef.current) d3.select(svgRef.current!).transition().duration(250).call(zoomRef.current.scaleBy, factor);
  };

  const findPath = async () => {
    if (!pathA.trim() || !pathB.trim()) return;
    setKGLoading(true);
    try {
      const res = await invokeTool("kg_path", { source: pathA.trim(), target: pathB.trim() });
      setPathResult(res.result ?? "No path found.");
      
      // Parse path result and highlight nodes/edges in the graph if path found
      if (res.result && !res.result.includes("No path") && !res.result.includes("not found")) {
        // Extract nodes from path result (format: "A —[rel]→ B —[rel]→ C")
        const pathNodes = res.result.match(/[\w\s]+(?=\s*—)/g);
        if (pathNodes && pathNodes.length > 0) {
          // TODO: Could highlight these nodes in the graph
          console.log("Path nodes:", pathNodes);
        }
      }
    } catch (e) {
      setPathResult("Error finding path.");
    } finally {
      setKGLoading(false);
    }
  };

  return (
    <div className="panel-wrap">
      <div className="panel-header">
        <GitBranch size={17} />
        <h2>Knowledge Graph</h2>
        <div className="panel-header-actions">
          <span className="stat-chip">{kgNodes.length} nodes</span>
          <span className="stat-chip">{kgEdges.length} edges</span>
          <span className="stat-chip">{orphanNodes.size} orphans</span>
          <button 
            className={`btn-icon ${viewMode === "graph" ? "active" : ""}`} 
            onClick={() => setViewMode("graph")}
            title="Graph View"
          >
            <Network size={13} />
          </button>
          <button 
            className={`btn-icon ${viewMode === "timeline" ? "active" : ""}`} 
            onClick={() => setViewMode("timeline")}
            title="Timeline View"
          >
            <Clock size={13} />
          </button>
          {viewMode === "graph" && (
            <>
              <button 
                className={`btn-icon ${!showOrphans ? "active" : ""}`}
                onClick={() => setShowOrphans(!showOrphans)}
                title={showOrphans ? "Hide Orphan Nodes" : "Show Orphan Nodes"}
              >
                {showOrphans ? "👁️" : "👁️‍🗨️"}
              </button>
              <button className="btn-icon" onClick={() => doZoom(1.4)}><ZoomIn size={13} /></button>
              <button className="btn-icon" onClick={() => doZoom(0.7)}><ZoomOut size={13} /></button>
            </>
          )}
          <button className="btn-icon" onClick={() => viewMode === "graph" ? loadGraph() : loadTimeline()}>
            <RefreshCw size={13} />
          </button>
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
        <button className="btn-small" onClick={findPath} disabled={kgLoading}>
          <Search size={11} /> {kgLoading ? "Searching..." : "Path"}
        </button>
      </div>
      {pathResult && (
        <div className="path-result-box">
          <div className="path-result-header">
            <span className="path-result-label">Path Found:</span>
            <button className="btn-icon-small" onClick={() => setPathResult("")} title="Clear">×</button>
          </div>
          <div className="path-result-content">{pathResult}</div>
        </div>
      )}

      {/* Graph */}
      <div className="kg-container">
        {viewMode === "graph" && (
          <>
            {kgLoading && <div className="kg-loading">Loading graph…</div>}
            {!kgLoading && kgNodes.length === 0 && (
              <div className="panel-empty">
                No graph data yet.<br />
                Create notes or use "Add to knowledge graph: [text]" to start building your graph.
              </div>
            )}
            <svg ref={svgRef} className="kg-svg" />
            {tooltip && (
              <div className="kg-tooltip" style={{ left: tooltip.x, top: tooltip.y }}>{tooltip.text}</div>
            )}
            
            {/* Node Details Popup */}
            {selectedNode && (
              <div className="node-details-overlay" onClick={() => setSelectedNode(null)}>
                <div className="node-details-popup" onClick={(e) => e.stopPropagation()}>
                  <div className="node-details-header">
                    <div>
                      <h3>{selectedNode.node.label}</h3>
                      <span className="node-type">{selectedNode.node.group}</span>
                    </div>
                    <button className="btn-icon" onClick={() => setSelectedNode(null)}>×</button>
                  </div>
                  
                  <div className="node-details-body">
                    <h4>Connections ({selectedNode.relations.length})</h4>
                    {selectedNode.relations.length === 0 ? (
                      <p className="no-connections">No connections yet</p>
                    ) : (
                      <div className="relations-list">
                        {selectedNode.relations.map((rel, idx) => (
                          <div key={idx} className="relation-item">
                            {"target" in rel ? (
                              // Outgoing relation
                              <span>
                                <strong>{selectedNode.node.label}</strong>
                                <span className="relation-arrow">—[{rel.relation}]→</span>
                                <strong>{rel.target}</strong>
                              </span>
                            ) : (
                              // Incoming relation
                              <span>
                                <strong>{rel.source}</strong>
                                <span className="relation-arrow">—[{rel.relation}]→</span>
                                <strong>{selectedNode.node.label}</strong>
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  
                  <div className="node-details-actions">
                    <button 
                      className="btn-small" 
                      onClick={() => {
                        expandNode(selectedNode.node.id);
                        setSelectedNode(null);
                      }}
                    >
                      Expand Neighbors
                    </button>
                    <button 
                      className="btn-small" 
                      onClick={() => {
                        setPathA(selectedNode.node.id);
                        setSelectedNode(null);
                      }}
                    >
                      Use in Path Finder
                    </button>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
        
        {viewMode === "timeline" && (
          <div className="timeline-view">
            <div className="timeline-info">
              <h3>Knowledge Graph Timeline</h3>
              <p>Shows recent nodes and when they were added to the graph.</p>
            </div>
            {timelineData ? (
              <pre className="timeline-content">{timelineData}</pre>
            ) : (
              <div className="panel-empty">
                Loading timeline data...<br />
                Note: Timeline shows when specific nodes were added. Use "Show timeline for [node name]" for detailed node evolution.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
