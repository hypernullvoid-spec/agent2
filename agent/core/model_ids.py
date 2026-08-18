"""
Model ID utilities.
"""

HUGGINGFACE_PREFIX = "huggingface/"


def strip_huggingface_model_prefix(model_id: str | None) -> str | None:
    """Remove 'huggingface/' prefix from model ID if present."""
    if not model_id:
        return None
    if model_id.startswith(HUGGINGFACE_PREFIX):
        return model_id[len(HUGGINGFACE_PREFIX):]
    return model_id