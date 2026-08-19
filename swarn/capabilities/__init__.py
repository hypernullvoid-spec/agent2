"""
Swarn capabilities — standalone, agent-registerable units of work.

  doc_intelligence   Visual Document Intelligence & Bounding-Box Inspector.
                     PDF/image → structured fields WITH visual coordinates →
                     annotated PNG + validated JSON. Registered with the agent
                     as the `swarn_doc_inspect` tool (see agent/tools.py) and
                     exposed on the CLI as `swarn doc-inspect`.

Nothing is imported eagerly here: pulling in `swarn.capabilities` must not
drag in Pillow, pydantic, or pdfplumber for a caller that only wanted a
different capability. Import the submodule you actually want.
"""

__all__ = ["doc_intelligence"]
