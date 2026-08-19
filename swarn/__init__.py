"""
`swarn` — capability packages that sit alongside the core `agent` package.

The historical layout puts everything the agent loop needs under `agent/`
(tools, phases 1–16). `swarn/` is for self-contained *capabilities*: units
of work with their own schemas, artifacts, and CLI surface that the agent
merely *registers* rather than depends on. A capability must be importable
and runnable on its own — `python examples/demo_doc_inspector.py` must work
without booting an agent loop, an LLM client, or a vector store.

The wiring direction is one-way and deliberate:

    agent/tools.py  ──imports──▶  swarn.capabilities.doc_intelligence

Never the reverse at import time. A capability may reuse an `agent.*` helper
(doc_intelligence borrows multimodal_rag's key/value line parser for its OCR
backend), but only via a lazy import inside the function that needs it, so a
missing optional dependency stays a runtime error string in one code path
instead of a startup crash for the whole agent.
"""

__all__ = ["capabilities"]
