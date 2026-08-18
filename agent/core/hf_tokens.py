"""
HF Token resolution.
"""

import os
from pathlib import Path


def resolve_hf_token() -> str | None:
    """Resolve HF token from environment or cache."""
    # Check environment variable first
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    # Check huggingface_hub cache
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token()
    except Exception:
        return None