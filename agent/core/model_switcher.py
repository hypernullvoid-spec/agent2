"""
Model switching utilities.
"""

from typing import Any
from agent.core.model_ids import strip_huggingface_model_prefix
from agent.core.local_models import is_local_model_id
from agent.utils.terminal_display import get_console, print_markdown


# Known valid model IDs for HF Router
VALID_MODEL_PREFIXES = (
    "zai-org/",
    "meta-llama/",
    "mistralai/",
    "google/",
    "microsoft/",
    "cohere/",
    "nvidia/",
    "databricks/",
    "ibm/",
    "tiiuae/",
    "Qwen/",
    "deepseek-ai/",
    "01-ai/",
    "baichuan-inc/",
    "THUDM/",
    "openchat/",
    "NousResearch/",
    "cognitivecomputations/",
    "mlabonne/",
    "mlx-community/",
    "unsloth/",
)


def is_valid_model_id(model_id: str) -> bool:
    """Check if a model ID is valid for HF Router."""
    if not model_id:
        return False
    normalized = strip_huggingface_model_prefix(model_id)
    if not normalized:
        return False

    # Check if it's a local model
    if is_local_model_id(normalized):
        return True

    # Check if it has a known HF Router prefix
    return any(normalized.startswith(prefix) for prefix in VALID_MODEL_PREFIXES)


def print_model_listing(config: Any, console) -> None:
    """Print available models."""
    console.print("\n[bold]Available Models (HF Router)[/bold]")
    console.print("[dim]Use '/model <id>' to switch. Prefix with 'huggingface/' for explicit routing.[/dim]\n")
    for prefix in VALID_MODEL_PREFIXES:
        console.print(f"  {prefix}*")


def print_invalid_id(model_id: str, console) -> None:
    """Print error for invalid model ID."""
    console.print(f"[bold red]Invalid model id:[/bold red] {model_id}")
    console.print("[dim]Use '/model' to see available models.[/dim]")


async def probe_and_switch_model(
    model_id: str,
    config: Any,
    session: Any,
    console,
    hf_token: str | None,
) -> bool:
    """Probe model and switch if valid."""
    if not is_valid_model_id(model_id):
        print_invalid_id(model_id, console)
        return False

    old_model = config.model_name
    config.model_name = model_id

    # If we have a session, we could probe here
    # For now, just update the config
    console.print(f"[green]Model switched:[/green] {old_model} → {model_id}")
    return True