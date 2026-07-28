"""
Phase 12: Tool Ecosystem & MCP Integration

Lets the agent connect to any Model Context Protocol (MCP) server —
GitHub, a database, a filesystem, a search API, anything that speaks the
protocol — and use its tools exactly like every tool already in
tools.py: same TOOL_REGISTRY, same run_tool() dispatch, same "errors are
strings, never raised" contract, same get_tool_definitions() that feeds
agent_loop.py and roles.py.

The async problem, and how this module solves it
───────────────────────────────────────────────────
The official MCP Python SDK (the `mcp` package) is async-only — every
operation (connect, list_tools, call_tool) is a coroutine. Every other
part of this codebase is synchronous: llm_client.py uses the blocking
Anthropic client, agent_loop.py's run() is a plain function, and
tools.py's run_tool() returns a plain string, not an awaitable. Making
the entire agent loop async just to support MCP would mean touching
every phase's code for one feature — exactly the kind of ripple effect
this codebase has avoided phase over phase (see tools.py's docstring:
"agent_loop.py needs zero changes").

Instead, this module runs ONE persistent asyncio event loop in a
dedicated background thread (MCPManager._loop_thread), started once at
first use and kept alive for the process's lifetime — MCP server
connections are stateful (they're a live subprocess + stdio pipe, or a
live SSE connection) and need a single consistent loop to run their
async generators correctly; you cannot just asyncio.run() a fresh loop
per call.

Every public, synchronous method here (connect_server, list_mcp_tools,
call_mcp_tool, disconnect_server) submits a coroutine to that background
loop via asyncio.run_coroutine_threadsafe() and blocks on the result
with a timeout. From the caller's side — including the dynamically
registered @tool wrappers this module creates — everything looks like
an ordinary synchronous function call. This is the standard bridge
pattern for "sync codebase needs to call into an async-only library";
it is not a partial or simplified version of it.

Dynamic tool registration
───────────────────────────
connect_server() doesn't just open a connection — it also calls
list_tools() on the server and registers ONE new entry in tools.py's
TOOL_REGISTRY per remote tool, named "mcp_<server_name>_<tool_name>" so
names from different servers can never collide. Each registered
function is a small closure that calls call_mcp_tool() with the right
server_name/tool_name baked in. This is why "any MCP server becomes
instantly usable" (per the original blueprint) — once connected, the
agent sees the remote tools in its normal tool list, with no special
casing anywhere else in the codebase. The Coder role in roles.py is the
one place an admin would add newly-registered MCP tool names if they
want a specific role to be allowed to use them (see roles.py's
docstring on tool allow-lists).

What this module does NOT do
────────────────────────────────
  - It does not include any specific MCP server implementation. You
    point it at a server command (stdio) the same way Claude Desktop's
    mcp config does — `connect_server("github", command="npx", args=[...])`.
  - It does not persist server configs across process restarts. That's
    a CLI/config-file concern (Phase 16's territory), not this module's.
  - It does not attempt SSE/HTTP transports in this version — only
    stdio, which covers the large majority of community MCP servers
    (anything you'd otherwise run with `npx <package>` or a local
    Python script). Adding SSE later is a transport-layer change inside
    connect_server, not a redesign of the bridge.
"""

import asyncio
import json
import threading
from typing import Optional

from agent.tools import TOOL_REGISTRY

DEFAULT_CALL_TIMEOUT_S = 60


class MCPManager:
    """
    Owns the background event loop and the set of live MCP server
    connections. One instance per process — same singleton pattern as
    every other manager class in this codebase (DataPipeline,
    FeatureEngine, ModelTrainer, ModelEvaluator, DeploymentPackager).
    """

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        # server_name -> {"session": ClientSession, "exit_stack": AsyncExitStack,
        #                  "tools": [tool_name, ...]}
        self._servers: dict[str, dict] = {}
        self._lock = threading.Lock()   # guards _servers and loop startup

    # ───────────────────────────────────────────────── background loop lifecycle

    def _ensure_loop_running(self) -> None:
        """
        Start the background event loop thread if it isn't already
        running. Idempotent and thread-safe — safe to call from every
        public method without callers needing to think about it.
        """
        with self._lock:
            if self._loop is not None and self._loop.is_running():
                return

            ready = threading.Event()

            def _run_loop():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                ready.set()
                loop.run_forever()

            self._loop_thread = threading.Thread(target=_run_loop, daemon=True, name="mcp-event-loop")
            self._loop_thread.start()
            ready.wait(timeout=5)

    def _run_coro(self, coro, timeout: float = DEFAULT_CALL_TIMEOUT_S):
        """
        The actual sync-to-async bridge: submit a coroutine to the
        background loop from whatever (synchronous) thread called us,
        and block until it finishes or times out. Every public method
        below is a thin wrapper around this.
        """
        self._ensure_loop_running()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ───────────────────────────────────────────────── connecting

    def connect_server(
        self,
        server_name: str,
        command: str,
        args: Optional[list[str]] = None,
        env: Optional[dict] = None,
    ) -> str:
        """
        Launch an MCP server as a subprocess (stdio transport), perform
        the MCP handshake, list its tools, and register each one as a
        normal tool in tools.py's TOOL_REGISTRY under the name
        "mcp_<server_name>_<tool_name>".

        command/args: same shape as Claude Desktop's mcp config, e.g.
          command="npx", args=["-y", "@modelcontextprotocol/server-github"]
        or for a local Python MCP server:
          command="python3", args=["my_server.py"]

        Implementation note: the actual connection (the stdio_client and
        ClientSession async context managers) is opened and held open
        by ONE dedicated background task that lives for the server's
        whole connected lifetime — see _server_task_main below. This is
        required, not a style choice: anyio (which the MCP SDK is built
        on) ties a context manager's cancel scope to the specific task
        that entered it, so a context opened in one coroutine cannot be
        closed by a *different* coroutine submitted later via
        run_coroutine_threadsafe, even though both run on the same event
        loop. Routing every tool call and the eventual disconnect
        through that one task's queue (rather than opening fresh
        coroutines per call) is what keeps everything in the same task
        for the server's whole lifetime.
        """
        if server_name in self._servers:
            return (
                f"Error: server '{server_name}' is already connected. "
                f"Call disconnect_server('{server_name}') first if you want to reconnect."
            )

        self._ensure_loop_running()

        request_queue: "asyncio.Queue" = asyncio.run_coroutine_threadsafe(
            self._make_queue(), self._loop
        ).result(timeout=5)
        ready_future = asyncio.run_coroutine_threadsafe(
            self._make_future(), self._loop
        ).result(timeout=5)

        task_future = asyncio.run_coroutine_threadsafe(
            self._server_task_main(server_name, command, args or [], env, request_queue, ready_future),
            self._loop,
        )

        try:
            connect_result = asyncio.run_coroutine_threadsafe(
                self._await_future(ready_future), self._loop
            ).result(timeout=30)
        except Exception as e:
            return f"Error connecting to MCP server '{server_name}': {type(e).__name__}: {e}"

        if isinstance(connect_result, Exception):
            return f"Error connecting to MCP server '{server_name}': {type(connect_result).__name__}: {connect_result}"

        tool_names, command_label = connect_result
        self._servers[server_name] = {
            "request_queue": request_queue,
            "task_future":   task_future,
            "tools":         tool_names,
            "command":       command_label,
        }

        tool_list = "\n".join(f"  {n}" for n in tool_names) or "  (no tools exposed)"
        return (
            f"Connected to MCP server '{server_name}' ({command_label}).\n"
            f"Registered {len(tool_names)} tool(s):\n{tool_list}\n"
            f"These are now available exactly like any other tool — call them directly, "
            f"or list them again with list_mcp_tools()."
        )

    async def _make_queue(self):
        return asyncio.Queue()

    async def _make_future(self):
        return self._loop.create_future()

    async def _await_future(self, future):
        return await future

    async def _server_task_main(self, server_name, command, args, env, request_queue, ready_future):
        """
        The ONE task that owns this server's connection for its entire
        lifetime. Opens the connection, signals readiness (or failure)
        back to connect_server() via ready_future, then loops forever
        servicing (tool_name, arguments, response_future) requests from
        request_queue until it receives a disconnect sentinel — at which
        point it closes the connection itself, in the same task that
        opened it, satisfying anyio's cancel-scope requirement.
        """
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from contextlib import AsyncExitStack

        exit_stack = AsyncExitStack()
        try:
            params = StdioServerParameters(command=command, args=args, env=env)
            read_stream, write_stream = await exit_stack.enter_async_context(stdio_client(params))
            session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()

            tools_result = await session.list_tools()
            tool_names = []
            for remote_tool in tools_result.tools:
                local_name = self._make_local_name(server_name, remote_tool.name)
                description = f"[MCP: {server_name}] {remote_tool.description or remote_tool.name}"
                schema = remote_tool.inputSchema or {"type": "object", "properties": {}}
                self._register_remote_tool(
                    server_name, remote_tool.name, local_name, request_queue, description, schema
                )
                tool_names.append(local_name)

        except Exception as e:
            ready_future.set_result(e)
            await exit_stack.aclose()
            return

        ready_future.set_result((tool_names, command))

        # ── service requests until told to stop, all within this same task ──
        try:
            while True:
                request = await request_queue.get()
                if request is None:   # disconnect sentinel
                    break
                tool_name, arguments, response_future = request
                try:
                    result = await session.call_tool(tool_name, arguments)
                    if result.isError:
                        text = self._extract_text(result.content)
                        response_future.set_result(f"Error from MCP tool '{tool_name}': {text}")
                    else:
                        response_future.set_result(self._extract_text(result.content))
                except Exception as e:
                    response_future.set_result(
                        f"Error calling MCP tool '{server_name}.{tool_name}': {type(e).__name__}: {e}"
                    )
        finally:
            await exit_stack.aclose()

    def _make_local_name(self, server_name: str, remote_tool_name: str) -> str:
        # Sanitize both halves so the combined name is always a valid,
        # collision-free Python identifier the @tool registry can key on.
        safe_server = "".join(c if c.isalnum() or c == "_" else "_" for c in server_name)
        safe_tool = "".join(c if c.isalnum() or c == "_" else "_" for c in remote_tool_name)
        return f"mcp_{safe_server}_{safe_tool}"

    def _register_remote_tool(
        self, server_name: str, remote_tool_name: str, local_name: str,
        request_queue, description: str, schema: dict,
    ) -> None:
        """
        Register one remote MCP tool into the SAME TOOL_REGISTRY every
        other tool in this codebase uses. This is the crux of "MCP tools
        become instantly usable, no special-casing" — agent_loop.py's
        get_tool_definitions()/run_tool() calls don't know or care that
        this particular entry proxies to a subprocess instead of running
        local Python.

        The registered closure submits a (tool_name, arguments,
        response_future) request onto this server's dedicated request
        queue rather than calling session.call_tool() directly — see
        connect_server's docstring for why the call must be serviced by
        the same task that opened the connection.
        """
        manager = self   # closed over below, avoids relying on the module-level singleton inside the closure

        def _call_this_remote_tool(**kwargs) -> str:
            return manager.call_mcp_tool(server_name, remote_tool_name, kwargs)

        _call_this_remote_tool.__name__ = local_name
        TOOL_REGISTRY[local_name] = {
            "description": description,
            "schema":      schema,
            "func":        _call_this_remote_tool,
        }

    # ───────────────────────────────────────────────── calling

    def call_mcp_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        """
        Invoke one tool on a connected MCP server and return its result
        as a string, formatted the same "never raise, return diagnostics
        as text" way every other tool in this codebase behaves. This is
        what the closures registered by connect_server() actually call.

        Puts a (tool_name, arguments, response_future) request onto the
        server's dedicated queue and waits for _server_task_main's loop
        — running in that server's one long-lived task — to service it
        and resolve the future. This indirection (queue + future) is
        what lets call_mcp_tool be invoked from any thread/task while
        the actual session.call_tool() always runs in the single task
        that owns the connection.
        """
        server = self._servers.get(server_name)
        if server is None:
            return (
                f"Error: MCP server '{server_name}' is not connected. "
                f"Connected servers: {list(self._servers.keys()) or '(none)'}"
            )
        try:
            result = self._run_coro(self._submit_call(server["request_queue"], tool_name, arguments))
        except asyncio.TimeoutError:
            return f"Error: MCP tool call to '{server_name}.{tool_name}' timed out after {DEFAULT_CALL_TIMEOUT_S}s."
        except Exception as e:
            return f"Error calling MCP tool '{server_name}.{tool_name}': {type(e).__name__}: {e}"
        return result

    async def _submit_call(self, request_queue, tool_name: str, arguments: dict) -> str:
        response_future = self._loop.create_future()
        await request_queue.put((tool_name, arguments, response_future))
        return await response_future

    @staticmethod
    def _extract_text(content_blocks) -> str:
        """
        MCP tool results are a list of content blocks (text/image/
        resource). Flatten the text ones into a single string — image/
        binary content isn't representable as the plain-string return
        type every tool in this codebase uses, so it's noted rather
        than silently dropped.
        """
        parts = []
        for block in content_blocks:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                parts.append(block.text)
            elif block_type == "image":
                parts.append("[MCP returned an image — not representable as text here]")
            elif block_type == "resource":
                parts.append(f"[MCP returned a resource: {getattr(block, 'resource', '?')}]")
            else:
                parts.append(f"[MCP returned unsupported content type: {block_type}]")
        return "\n".join(parts) if parts else "(empty result)"

    # ───────────────────────────────────────────────── inspection & teardown

    def list_mcp_servers(self) -> str:
        if not self._servers:
            return "No MCP servers connected. Use connect_server(name, command, args) first."
        lines = ["Connected MCP servers:"]
        for name, info in self._servers.items():
            lines.append(f"  {name}  ({info['command']})  — {len(info['tools'])} tool(s)")
        return "\n".join(lines)

    def list_mcp_tools(self, server_name: Optional[str] = None) -> str:
        if not self._servers:
            return "No MCP servers connected."
        lines = ["MCP-provided tools:"]
        for name, info in self._servers.items():
            if server_name and name != server_name:
                continue
            for tool_name in info["tools"]:
                lines.append(f"  {tool_name}  (server: {name})")
        return "\n".join(lines)

    def disconnect_server(self, server_name: str) -> str:
        """
        Signal the server's dedicated task to stop (via the None
        sentinel on its request queue) and wait for that task to finish
        — which is when it actually closes the connection, in the same
        task that opened it. disconnect_server does NOT close the
        connection itself; it only asks the owning task to do so, which
        is the fix for the cancel-scope violation a more direct
        "just call aclose() from here" implementation would hit.
        """
        server = self._servers.get(server_name)
        if server is None:
            return f"Error: MCP server '{server_name}' is not connected."

        try:
            self._run_coro(server["request_queue"].put(None))
            # Wait for _server_task_main to actually finish (it closes
            # the exit_stack in its own `finally`, in its own task).
            server["task_future"].result(timeout=15)
        except Exception as e:
            return f"Error closing MCP server '{server_name}': {type(e).__name__}: {e}"

        for tool_name in server["tools"]:
            TOOL_REGISTRY.pop(tool_name, None)
        del self._servers[server_name]
        return f"Disconnected '{server_name}' and removed its {len(server['tools'])} tool(s) from the registry."

    def shutdown(self) -> None:
        """Close every connection and stop the background loop — call on process exit."""
        for name in list(self._servers.keys()):
            self.disconnect_server(name)
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)


# ─── singleton, matching the rest of the codebase ──────────────────────────────

_manager: Optional[MCPManager] = None


def get_mcp_manager() -> MCPManager:
    global _manager
    if _manager is None:
        _manager = MCPManager()
    return _manager
