"""
Local model detection.
"""

LOCAL_MODEL_PREFIXES = (
    "ollama/",
    "lmstudio/",
    "llama.cpp/",
    "gguf:",
    "local:",
)


def is_local_model_id(model_id: str | None) -> bool:
    """Check if a model ID refers to a local model."""
    if not model_id:
        return False
    return any(model_id.startswith(prefix) for prefix in LOCAL_MODEL_PREFIXES)