"""
app/tools/mcp_client.py

MCP (Model Context Protocol) Client

Connects to any MCP server over HTTP/SSE or stdio.
Discovers tools dynamically, wraps them as async callables,
and registers them into TOOL_REGISTRY.

After registration you immediately gain access to whatever
the connected server exposes — Gmail, Google Calendar, GitHub,
browser automation, databases, etc. — with zero custom code.

Supported transports:
  - HTTP/SSE  (most hosted servers, e.g. mcp.asana.com)
  - stdio     (local servers — child process with JSON-RPC over stdin/stdout)

Configuration via .env or environment variables:

    MCP_SERVERS='[
        {"name": "gmail",     "transport": "http", "url": "http://localhost:3001/mcp"},
        {"name": "calendar",  "transport": "http", "url": "http://localhost:3002/mcp"},
        {"name": "github",    "transport": "stdio", "command": "npx @modelcontextprotocol/server-github"}
    ]'

Integration — call at startup (in main.py, after settings are loaded):

    from app.tools.mcp_client import mcp_registry
    await mcp_registry.connect_all()

"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger


# ─── Data models ─────────────────────────────────────────────────────────────

@dataclass
class MCPTool:
    """A single tool exposed by an MCP server."""
    name: str                          # original name from server
    registry_name: str                 # prefixed name in TOOL_REGISTRY
    description: str
    input_schema: dict                 # JSON Schema for parameters
    server_name: str                   # which server this came from


@dataclass
class MCPServer:
    """Configuration for a single MCP server."""
    name: str
    transport: str                     # "http" or "stdio"
    url: Optional[str] = None          # for http transport
    command: Optional[str] = None      # for stdio transport
    env: dict[str, str] = field(default_factory=dict)

    # Runtime state
    _process: Optional[Any] = field(default=None, repr=False)
    _stdin: Optional[Any] = field(default=None, repr=False)
    _stdout: Optional[Any] = field(default=None, repr=False)
    _stderr_task: Optional[Any] = field(default=None, repr=False)
    _http_session: Optional[Any] = field(default=None, repr=False)
    _next_id: int = field(default=1, repr=False)
    # ponytail: per-server lock serialises stdio requests; add a reader task
    # with an id→future map if concurrent tool calls ever matter.
    _io_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


# ─── JSON-RPC helpers ─────────────────────────────────────────────────────────

def _rpc_request(method: str, params: dict, id: int) -> dict:
    return {"jsonrpc": "2.0", "method": method, "params": params, "id": id}


def _rpc_notify(method: str, params: dict) -> dict:
    return {"jsonrpc": "2.0", "method": method, "params": params}


# ─── MCP Registry ─────────────────────────────────────────────────────────────

class MCPRegistry:
    """
    Manages connections to multiple MCP servers and provides a unified
    interface for tool discovery and execution.
    """

    def __init__(self):
        self._servers: dict[str, MCPServer] = {}
        self._tools: dict[str, MCPTool] = {}        # registry_name → MCPTool
        self._server_tools: dict[str, list[str]] = {}  # server_name → [registry_names]
        self._connected: set[str] = set()

    # ── Configuration ──────────────────────────────────────────────────────

    def configure(self, servers: list[dict]) -> None:
        """
        Load server configs. Call this once at startup before connect_all().

        servers: list of dicts with keys:
            name, transport, url (http) or command (stdio), env (optional)
        """
        for cfg in servers:
            name = cfg["name"]
            self._servers[name] = MCPServer(
                name=name,
                transport=cfg.get("transport", "http"),
                url=cfg.get("url"),
                command=cfg.get("command"),
                env=cfg.get("env", {}),
            )
            logger.info("MCP: configured server '{}' ({})", name, cfg.get("transport", "http"))

    def configure_from_env(self) -> None:
        """Load server config from MCP_SERVERS env var (JSON array)."""
        from app.config import settings
        raw = settings.mcp_servers or "[]"
        try:
            servers = json.loads(raw)
            self.configure(servers)
        except json.JSONDecodeError as e:
            logger.error("MCP_SERVERS env var is not valid JSON: {}", e)

    # ── Connection lifecycle ────────────────────────────────────────────────

    async def connect_all(self) -> dict[str, bool]:
        """
        Connect to all configured servers and discover their tools.
        Returns {server_name: success} for each server.
        """
        results = {}
        tasks = [self._connect_server(name) for name in self._servers]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        for name, outcome in zip(self._servers, outcomes):
            ok = not isinstance(outcome, Exception)
            results[name] = ok
            if not ok:
                logger.error("MCP: failed to connect to '{}': {}", name, outcome)
        return results

    async def _connect_server(self, name: str) -> None:
        server = self._servers[name]
        if server.transport == "http":
            await self._connect_http(server)
        elif server.transport == "stdio":
            await self._connect_stdio(server)
        else:
            raise ValueError(f"Unknown transport: {server.transport}")

        await self._discover_tools(server)
        self._connected.add(name)
        logger.info(
            "MCP: connected to '{}' — {} tools discovered",
            name,
            len(self._server_tools.get(name, [])),
        )

    async def _connect_http(self, server: MCPServer) -> None:
        """Establish HTTP session and send MCP initialize."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required for HTTP MCP transport: pip install httpx")

        server._http_session = httpx.AsyncClient(
            base_url=server.url,
            timeout=httpx.Timeout(30.0),
            headers={"Content-Type": "application/json"},
        )

        # MCP initialize handshake
        resp = await server._http_session.post(
            "",
            json=_rpc_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "jarvis", "version": "1.0"},
                },
                id=server._next_id,
            ),
        )
        resp.raise_for_status()
        server._next_id += 1

        # Send initialized notification
        await server._http_session.post(
            "", json=_rpc_notify("notifications/initialized", {})
        )

    async def _connect_stdio(self, server: MCPServer) -> None:
        """Launch child process and set up stdio JSON-RPC."""
        if not server.command:
            raise ValueError(f"stdio server '{server.name}' has no command")

        import asyncio

        env = {**os.environ, **server.env}
        proc = await asyncio.create_subprocess_shell(
            server.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        server._process = proc
        server._stdin   = proc.stdin
        server._stdout  = proc.stdout

        # stderr must be drained continuously. Many stdio servers log heavily
        # there, and a full pipe buffer would block the child process forever.
        async def _drain_stderr() -> None:
            try:
                while True:
                    line = await proc.stderr.readline()
                    if not line:
                        break
                    logger.debug(
                        "MCP '{}' stderr: {}",
                        server.name, line.decode(errors="replace").rstrip(),
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("MCP '{}' stderr reader stopped: {}", server.name, exc)

        server._stderr_task = asyncio.create_task(_drain_stderr())

        # Initialize
        await self._stdio_request(server, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "jarvis", "version": "1.0"},
        })
        await self._stdio_notify(server, "notifications/initialized", {})

    # ── Tool discovery ──────────────────────────────────────────────────────

    async def _discover_tools(self, server: MCPServer) -> None:
        """Fetch tools/list from the server and register them."""
        try:
            result = await self._call(server, "tools/list", {})
            tools = result.get("tools", [])
        except Exception as e:
            logger.warning("MCP: tools/list failed for '{}': {}", server.name, e)
            return

        registered = []
        for tool in tools:
            registry_name = f"mcp_{server.name}_{tool['name']}"
            mcp_tool = MCPTool(
                name=tool["name"],
                registry_name=registry_name,
                description=tool.get("description", ""),
                input_schema=tool.get("inputSchema", {}),
                server_name=server.name,
            )
            self._tools[registry_name] = mcp_tool
            registered.append(registry_name)

            # Register in TOOL_REGISTRY
            self._register_tool(mcp_tool)

        self._server_tools[server.name] = registered

    def _register_tool(self, mcp_tool: MCPTool) -> None:
        """Add this MCP tool to the main TOOL_REGISTRY and TOOL_SCHEMA."""
        from app.tools.router import TOOL_REGISTRY, TOOL_SCHEMA

        # Create a callable that dispatches through this registry
        async def _caller(**kwargs):
            return await self.call_tool(mcp_tool.registry_name, kwargs)

        TOOL_REGISTRY[mcp_tool.registry_name] = _caller

        # Build schema from JSON Schema (extract required property names)
        required = mcp_tool.input_schema.get("required", [])
        TOOL_SCHEMA[mcp_tool.registry_name] = required

        logger.debug("MCP: registered tool '{}'", mcp_tool.registry_name)

    # ── Tool execution ──────────────────────────────────────────────────────

    async def call_tool(self, registry_name: str, params: dict) -> str:
        """
        Call an MCP tool by its registry name.
        Returns string result suitable for the tool router.
        """
        if registry_name not in self._tools:
            return f"MCP tool '{registry_name}' not found."

        mcp_tool = self._tools[registry_name]
        server = self._servers.get(mcp_tool.server_name)
        if not server or mcp_tool.server_name not in self._connected:
            return f"MCP server '{mcp_tool.server_name}' is not connected."

        try:
            result = await self._call(server, "tools/call", {
                "name": mcp_tool.name,
                "arguments": params,
            })
            return self._format_tool_result(result)
        except Exception as e:
            logger.error("MCP tool '{}' failed: {}", registry_name, e)
            return f"MCP tool error: {e}"

    @staticmethod
    def _format_tool_result(result: dict) -> str:
        """Convert MCP tool result to a plain string."""
        content = result.get("content", [])
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") == "resource":
                parts.append(str(item.get("resource", "")))
        return "\n".join(parts) or str(result)

    # ── RPC transport layer ──────────────────────────────────────────────────

    async def _call(self, server: MCPServer, method: str, params: dict) -> dict:
        """Route to correct transport."""
        if server.transport == "http":
            return await self._http_request(server, method, params)
        elif server.transport == "stdio":
            return await self._stdio_request(server, method, params)
        raise ValueError(f"Unknown transport: {server.transport}")

    async def _http_request(self, server: MCPServer, method: str, params: dict) -> dict:
        req_id = server._next_id
        server._next_id += 1
        resp = await server._http_session.post(
            "", json=_rpc_request(method, params, req_id)
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")
        return data.get("result", {})

    async def _stdio_request(self, server: MCPServer, method: str, params: dict) -> dict:
        async with server._io_lock:
            req_id = server._next_id
            server._next_id += 1
            line = json.dumps(_rpc_request(method, params, req_id)) + "\n"
            server._stdin.write(line.encode())
            await server._stdin.drain()
            # Servers may emit notifications (no id) before the reply; read
            # until the message with our id arrives. EOF means the child died.
            while True:
                raw = await server._stdout.readline()
                if not raw:
                    raise RuntimeError(f"MCP server '{server.name}' closed stdout")
                try:
                    data = json.loads(raw.decode())
                except json.JSONDecodeError:
                    logger.debug("MCP '{}': non-JSON line on stdout: {}", server.name, raw[:120])
                    continue
                if data.get("id") == req_id:
                    break
                logger.debug("MCP '{}': skipping message {}", server.name, data.get("method", data.get("id")))
        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")
        return data.get("result", {})

    async def _stdio_notify(self, server: MCPServer, method: str, params: dict) -> None:
        line = json.dumps(_rpc_notify(method, params)) + "\n"
        server._stdin.write(line.encode())
        await server._stdin.drain()

    # ── Status ───────────────────────────────────────────────────────────────

    def status(self) -> str:
        """Return a human-readable status string."""
        if not self._servers:
            return (
                "MCP: no servers configured.\n"
                "Add servers via MCP_SERVERS env var or mcp_registry.configure([...])."
            )

        lines = [f"MCP Servers ({len(self._servers)} configured, {len(self._connected)} connected):"]
        for name, server in self._servers.items():
            connected = "✓" if name in self._connected else "✗"
            tool_count = len(self._server_tools.get(name, []))
            lines.append(
                f"  {connected} {name} ({server.transport}) — {tool_count} tools"
            )

        if self._tools:
            lines.append(f"\nRegistered MCP tools ({len(self._tools)}):")
            for rname, tool in list(self._tools.items())[:20]:
                lines.append(f"  {rname}: {tool.description[:60]}")
            if len(self._tools) > 20:
                lines.append(f"  … and {len(self._tools)-20} more")

        return "\n".join(lines)

    def list_tools(self) -> list[str]:
        """Return all registered MCP tool registry names."""
        return list(self._tools.keys())

    async def disconnect_all(self) -> None:
        """Clean up connections."""
        for name, server in self._servers.items():
            try:
                if server._http_session:
                    await server._http_session.aclose()
                if server._stderr_task:
                    server._stderr_task.cancel()
                    try:
                        await server._stderr_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    server._stderr_task = None
                if server._process:
                    server._process.terminate()
                    await server._process.wait()
            except Exception as e:
                logger.warning("MCP: error disconnecting '{}': {}", name, e)
        self._connected.clear()


# ─── Module-level singleton ───────────────────────────────────────────────────

mcp_registry = MCPRegistry()


# ─── Tool functions for the TOOL_REGISTRY ────────────────────────────────────

async def mcp_status() -> str:
    """Show the status of all configured MCP servers and their tools."""
    return mcp_registry.status()


async def mcp_list_tools() -> str:
    """List all tools currently available through MCP servers."""
    tools = mcp_registry.list_tools()
    if not tools:
        return "No MCP tools registered. Connect an MCP server first."
    return "Available MCP tools:\n" + "\n".join(f"  - {t}" for t in sorted(tools))
