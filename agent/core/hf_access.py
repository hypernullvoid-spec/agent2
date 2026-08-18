"""
HF Access - fetch user info from Hugging Face.
"""

from typing import Any


async def fetch_whoami_v2(token: str | None) -> dict[str, Any] | None:
    """Fetch user info from HF whoami v2 endpoint."""
    if not token:
        return None
    try:
        from huggingface_hub import HfApi
        import asyncio

        api = HfApi(token=token)
        return await asyncio.to_thread(api.whoami)
    except Exception:
        return None


def normalize_hf_user_plan(whoami: dict[str, Any] | None) -> str | None:
    """Normalize HF user plan from whoami response."""
    if not whoami or not isinstance(whoami, dict):
        return None

    # Check for plan info
    plan = whoami.get("plan")
    if isinstance(plan, str):
        return plan.lower()

    # Check for tier
    tier = whoami.get("tier")
    if isinstance(tier, str):
        return tier.lower()

    # Check for subscription
    subscription = whoami.get("subscription")
    if isinstance(subscription, dict):
        plan = subscription.get("plan")
        if isinstance(plan, str):
            return plan.lower()

    return None