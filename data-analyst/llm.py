"""
One interface for every provider.

Set whichever key you have. The code does not change.

    export ANTHROPIC_API_KEY=sk-ant-...     -> uses Claude
    export GROQ_API_KEY=gsk_...             -> uses Llama on Groq
    export OPENAI_API_KEY=sk-...            -> uses GPT

If several keys are set, pick one explicitly:

    export LLM_PROVIDER=groq

Everything else in the project calls llm.chat() and never knows the difference.
"""

import os

# ----------------------------------------------------------------------
# WHAT EACH PROVIDER OFFERS
# ----------------------------------------------------------------------
#
# small   - cheap and fast. Used for picking columns and groupings.
# large   - does the actual analysis and writes the answer.
# budget  - how many tokens of DATA it is safe to send in one question.
#           This is the number that differs most between providers, and it is
#           not the context window: on Groq's free tier the real limit is
#           100,000 tokens PER DAY, so we budget far below the window.
#
# Override any model with an env var, e.g. LLM_LARGE=llama-3.1-8b-instant

PROVIDERS = {
    "anthropic": {
        "key_env": "ANTHROPIC_API_KEY",
        "small":   "claude-haiku-4-5",
        "large":   "claude-opus-5",
        "budget":  300_000,
        "package": "anthropic",
    },
    "groq": {
        "key_env":  "GROQ_API_KEY",
        "small":    "llama-3.1-8b-instant",
        "large":    "llama-3.3-70b-versatile",
        "budget":   12_000,
        "package":  "groq",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "openai": {
        "key_env": "OPENAI_API_KEY",
        "small":   "gpt-4o-mini",
        "large":   "gpt-4o",
        "budget":  100_000,
        "package": "openai",
    },
}

# Checked in this order when no LLM_PROVIDER is set
ORDER = ["anthropic", "openai", "groq"]

_provider = None
_client = None


# ----------------------------------------------------------------------
# WHICH PROVIDER ARE WE USING?
# ----------------------------------------------------------------------

def provider():
    """Work out the provider once, from whichever key is present."""
    global _provider
    if _provider:
        return _provider

    chosen = os.environ.get("LLM_PROVIDER", "").strip().lower()

    if chosen:
        if chosen not in PROVIDERS:
            raise ValueError(f"LLM_PROVIDER={chosen!r}. Choose one of: {', '.join(PROVIDERS)}")
        if not os.environ.get(PROVIDERS[chosen]["key_env"]):
            raise ValueError(f"LLM_PROVIDER={chosen} but {PROVIDERS[chosen]['key_env']} is not set.")
    else:
        chosen = next((p for p in ORDER if os.environ.get(PROVIDERS[p]["key_env"])), None)
        if not chosen:
            raise ValueError(
                "No API key found. Set one of:\n"
                + "\n".join(f"  export {c['key_env']}=..."
                            for c in PROVIDERS.values())
            )

    _provider = chosen
    return chosen


def config():
    """The active provider's settings, with any env-var overrides applied."""
    settings = dict(PROVIDERS[provider()])
    settings["small"]  = os.environ.get("LLM_SMALL",  settings["small"])
    settings["large"]  = os.environ.get("LLM_LARGE",  settings["large"])
    settings["budget"] = int(os.environ.get("LLM_BUDGET", settings["budget"]))
    return settings


def budget():
    """How many tokens of data it is safe to send. Ask this instead of guessing."""
    return config()["budget"]


def info():
    """One line describing what is active. Print this at startup."""
    c = config()
    return f"{provider()}  (small={c['small']}, large={c['large']}, budget={c['budget']:,} tokens)"


# ----------------------------------------------------------------------
# THE CLIENT
# ----------------------------------------------------------------------

def _get_client():
    global _client
    if _client is not None:
        return _client

    name, settings = provider(), config()

    try:
        if name == "anthropic":
            import anthropic
            _client = anthropic.Anthropic()
        else:
            # Groq and OpenAI both speak the OpenAI chat format, so one SDK
            # covers both - Groq just needs a different base_url.
            from openai import OpenAI
            _client = OpenAI(
                api_key=os.environ[settings["key_env"]],
                base_url=settings.get("base_url"),
            )
    except ImportError as error:
        raise ImportError(
            f"Install the SDK for {name}:  pip install "
            + ("anthropic" if name == "anthropic" else "openai")
        ) from error

    return _client


# ----------------------------------------------------------------------
# THE ONE FUNCTION EVERYTHING ELSE CALLS
# ----------------------------------------------------------------------

def chat(prompt, tier="large", max_tokens=1500, temperature=0):
    """
    Send a prompt, get plain text back.

    tier: "small" for cheap mechanical jobs, "large" for the real analysis.
    """
    settings = config()
    model = settings[tier]

    if provider() == "anthropic":
        reply = _get_client().messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in reply.content if b.type == "text").strip()

    reply = _get_client().chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return (reply.choices[0].message.content or "").strip()


def count_tokens(text):
    """
    Exact where the provider offers it, conservative estimate otherwise.

    The estimate divides by 2.8 rather than 3.5 on purpose. Measured against a
    real count, 3.5 came out about 25% LOW - and being low is the dangerous
    direction, because you believe it fits and the API rejects it.
    """
    if provider() == "anthropic":
        try:
            return _get_client().messages.count_tokens(
                model=config()["large"],
                messages=[{"role": "user", "content": text}],
            ).input_tokens
        except Exception:
            pass                      # fall through to the estimate

    return int(len(text) / 2.8)


def reset():
    """Forget the cached provider - useful in tests that swap env vars."""
    global _provider, _client
    _provider, _client = None, None


if __name__ == "__main__":
    print("Active provider:", info())
    print()
    print("Test call:", chat("Reply with exactly: OK", tier="small", max_tokens=10))
