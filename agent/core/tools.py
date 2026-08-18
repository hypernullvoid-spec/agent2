"""
Tool Router for MCP and local tools.
"""

from typing import Any
from agent.runtime.tools import get_tool_definitions, run_tool


class ToolRouter:
    """Routes tool calls to local or MCP tools."""

    def __init__(
        self,
        mcp_servers: dict[str, Any],
        hf_token: str | None = None,
        local_mode: bool = True,
    ):
        self.mcp_servers = mcp_servers
        self.hf_token = hf_token
        self.local_mode = local_mode
        self._mcp_manager = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._mcp_manager:
            await self._mcp_manager.disconnect_all()

    def get_tool_definitions(self, tool_names: list[str] | None = None) -> list[dict]:
        """Get tool definitions in Anthropic format."""
        return get_tool_definitions(tool_names)

    async def run_tool(self, name: str, tool_input: dict) -> str:
        """Execute a tool by name."""
        return run_tool(name, tool_input)