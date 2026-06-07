/**
 * src/components/DocsPanel.tsx
 * RAG document management — ingest, list, search, remove.
 */
import React, { useEffect, useState } from "react";
import { FileText, RefreshCw, Trash2, Search, FolderOpen, Plus } from "lucide-react";
import { useStore } from "../stores/useStore";
import { listDocuments, ingestDocument, ingestDirectory, removeDocument, docSearch, unifiedSearch } from "../api";
import type { RagDoc } from "../api";

function parseDocList(raw: string): RagDoc[] {
  // Format: "  [PDF] Title — N chunks (indexed YYYY-MM-DD)"
  const docs: RagDoc[] = [];
  const re = /\[([A-Z]+)\]\s+(.+?)\s+—\s+(\d+)\s+chunk.+indexed\s+([\d-]+)/g;
  let m;
  while ((m = re.exec(raw)) !== null) {
    docs.push({ title: m[2].trim(), source_path: "", chunks: parseInt(m[3]), last_indexed: m[4] });
  }
  return docs;
}

function DocRow({ doc, onRemove }: { doc: RagDoc; onRemove: () => void }) {
  const ext = doc.title.split(".").pop()?.toUpperCase() ?? "DOC";
  return (
    <div className="doc-row">
      <div className="doc-ext-badge">{ext}</div>
      <div className="doc-info">
        <div className="doc-title">{doc.title}</div>
        <div className="doc-meta">{doc.chunks} chunks · {doc.last_indexed}</div>
      </div>
      <button className="btn-icon delete-btn" onClick={onRemove} title="Remove">
        <Trash2 size={12} />
      </button>
    </div>
  );
}

export function DocsPanel() {
  const { ragDocs, ragDocsLoading, setRagDocs, setRagDocsLoading } = useStore();
  const [rawText, setRawText]     = useState("");
  const [ingestPath, setIngestPath] = useState("");
  const [ingestType, setIngestType] = useState<"file" | "dir">("file");
  const [ingesting, setIngesting] = useState(false);
  const [ingestResult, setIngestResult] = useState("");
  const [query, setQuery]         = useState("");
  const [searchResult, setSearchResult] = useState("");
  const [searching, setSearching] = useState(false);
  const [searchMode, setSearchMode] = useState<"docs" | "unified">("docs");

  const load = async () => {
    setRagDocsLoading(true);
    try {
      const res = await listDocuments();
      setRawText(res.result ?? "");
      setRagDocs(parseDocList(res.result ?? ""));
    } catch { /* ignore */ } finally { setRagDocsLoading(false); }
  };

  useEffect(() => { load(); }, []);

  const handleIngest = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!ingestPath.trim()) return;
    setIngesting(true); setIngestResult("");
    try {
      const res = ingestType === "file"
        ? await ingestDocument(ingestPath.trim())
        : await ingestDirectory(ingestPath.trim());
      setIngestResult(res.result ?? "Done.");
      setIngestPath("");
      load();
    } catch (err) {
      setIngestResult(String(err));
    } finally { setIngesting(false); }
  };

  const handleRemove = async (path: string, title: string) => {
    await removeDocument(path || title);
    load();
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setSearching(true); setSearchResult("");
    try {
      const res = searchMode === "unified"
        ? await unifiedSearch(query.trim())
        : await docSearch(query.trim());
      setSearchResult(res.result ?? "No results.");
    } catch { /* ignore */ } finally { setSearching(false); }
  };

  return (
    <div className="panel-wrap">
      <div className="panel-header">
        <FileText size={17} />
        <h2>Documents</h2>
        <button className="btn-icon" onClick={load}><RefreshCw size={13} /></button>
      </div>

      {/* Ingest form */}
      <div className="docs-section">
        <div className="docs-section-label">Ingest</div>
        <form className="ingest-form" onSubmit={handleIngest}>
          <div className="ingest-type-toggle">
            <button type="button"
              className={`ingest-type-btn ${ingestType === "file" ? "active" : ""}`}
              onClick={() => setIngestType("file")}>
              <FileText size={11} /> File
            </button>
            <button type="button"
              className={`ingest-type-btn ${ingestType === "dir" ? "active" : ""}`}
              onClick={() => setIngestType("dir")}>
              <FolderOpen size={11} /> Directory
            </button>
          </div>
          <input
            className="search-input ingest-input"
            value={ingestPath}
            onChange={e => setIngestPath(e.target.value)}
            placeholder={ingestType === "file" ? "/path/to/document.pdf" : "/path/to/folder/"}
          />
          <button type="submit" className="btn-small" disabled={ingesting || !ingestPath.trim()}>
            {ingesting ? "Ingesting…" : <><Plus size={11} /> Ingest</>}
          </button>
        </form>
        {ingestResult && <div className="ingest-result">{ingestResult}</div>}
      </div>

      {/* Search */}
      <div className="docs-section">
        <div className="docs-section-label">
          Search
          <div className="search-mode-toggle">
            <button className={`ingest-type-btn ${searchMode === "docs" ? "active" : ""}`}
              onClick={() => setSearchMode("docs")}>Docs only</button>
            <button className={`ingest-type-btn ${searchMode === "unified" ? "active" : ""}`}
              onClick={() => setSearchMode("unified")}>Docs + Notes</button>
          </div>
        </div>
        <form className="search-bar" onSubmit={handleSearch}>
          <Search size={13} className="search-icon" />
          <input
            className="search-input"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="What did the contract say about…"
          />
          <button type="submit" className="btn-small" disabled={searching || !query.trim()}>
            {searching ? "…" : "Search"}
          </button>
        </form>
        {searchResult && <pre className="search-result-box">{searchResult}</pre>}
      </div>

      {/* Document list */}
      <div className="docs-section docs-list-section">
        <div className="docs-section-label">{ragDocs.length} document{ragDocs.length !== 1 ? "s" : ""} indexed</div>
        {ragDocsLoading ? (
          <div className="panel-loading">Loading…</div>
        ) : ragDocs.length === 0 ? (
          <div className="panel-empty small">No documents ingested yet.</div>
        ) : (
          <div className="doc-list">
            {ragDocs.map((d, i) => (
              <DocRow key={i} doc={d} onRemove={() => handleRemove(d.source_path, d.title)} />
            ))}
          </div>
        )}
        {rawText && ragDocs.length === 0 && (
          <pre className="stats-pre" style={{ marginTop: 8 }}>{rawText}</pre>
        )}
      </div>
    </div>
  );
}
