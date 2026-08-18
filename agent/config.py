"""
CLI Agent Configuration
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import json
from pathlib import Path


@dataclass
class MCPServerConfig:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class MessagingConfig:
    enabled: bool = False
    webhook_url: Optional[str] = None
    default_destinations: list[str] = field(default_factory=list)

    def default_auto_destinations(self) -> list[str]:
        return self.default_destinations


def _default_model_name() -> str:
    # Imported lazily: agent.llm.router reads env at import time and pulls in
    # the client stack, which config.py callers shouldn't pay for.
    from agent.llm import DEPLOYED_MODEL_NAME
    return DEPLOYED_MODEL_NAME


@dataclass
class CLIConfig:
    model_name: str = field(default_factory=_default_model_name)
    reasoning_effort: Optional[str] = None
    yolo_mode: bool = False
    tool_runtime: str = "local"
    max_iterations: int = 50
    mcpServers: dict[str, MCPServerConfig] = field(default_factory=dict)
    messaging: MessagingConfig = field(default_factory=MessagingConfig)
    share_traces: bool = False
    personal_trace_repo_template: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serializable view — the inverse of load_config()."""
        return {
            "model_name": self.model_name,
            "reasoning_effort": self.reasoning_effort,
            "yolo_mode": self.yolo_mode,
            "tool_runtime": self.tool_runtime,
            "max_iterations": self.max_iterations,
            "mcpServers": {
                name: {"command": s.command, "args": s.args, "env": s.env}
                for name, s in self.mcpServers.items()
            },
            "messaging": {
                "enabled": self.messaging.enabled,
                "webhook_url": self.messaging.webhook_url,
                "default_destinations": self.messaging.default_destinations,
            },
            "share_traces": self.share_traces,
            "personal_trace_repo_template": self.personal_trace_repo_template,
        }


DEFAULT_CONFIG_DIR = Path.home() / ".config" / "swarn"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "cli_agent_config.json"


def load_config(config_path: Optional[Path] = None, include_user_defaults: bool = True) -> CLIConfig:
    """Load configuration from JSON file (defaults to DEFAULT_CONFIG_PATH)."""
    config_path = config_path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return CLIConfig()

    try:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        # A hand-edited or half-written config must not stop the CLI from
        # starting — fall back to defaults.
        return CLIConfig()

    mcp_servers = {}
    for name, server_data in data.get("mcpServers", {}).items():
        mcp_servers[name] = MCPServerConfig(
            command=server_data.get("command", ""),
            args=server_data.get("args", []),
            env=server_data.get("env", {}),
        )

    messaging_data = data.get("messaging", {})
    messaging = MessagingConfig(
        enabled=messaging_data.get("enabled", False),
        webhook_url=messaging_data.get("webhook_url"),
        default_destinations=messaging_data.get("default_destinations", []),
    )

    return CLIConfig(
        model_name=data.get("model_name") or _default_model_name(),
        reasoning_effort=data.get("reasoning_effort"),
        yolo_mode=data.get("yolo_mode", False),
        tool_runtime=data.get("tool_runtime", "local"),
        max_iterations=data.get("max_iterations", 50),
        mcpServers=mcp_servers,
        messaging=messaging,
        share_traces=data.get("share_traces", False),
        personal_trace_repo_template=data.get("personal_trace_repo_template"),
    )


def save_config(config: CLIConfig, config_path: Optional[Path] = None) -> Path:
    """Persist `config` as JSON so /model, /yolo, /effort and /share-traces
    survive across sessions. Returns the path written."""
    config_path = config_path or DEFAULT_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
    return config_path