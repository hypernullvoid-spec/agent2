"""
Task routing for `swarn run` — the universal entry point.

`swarn run` accepts an arbitrary task. Some of those tasks are really just
"answer this question about this document", which `swarn ask` already does
end-to-end: parse once, transcribe with line ids, verify every quote against
the document, re-evaluate the arithmetic locally, box the evidence.

Sending that through the ReAct agent instead would cost three or more LLM
calls to reach the same tool — and, worse, the agent would then *paraphrase*
the verified JSON in its own words. That paraphrase sits after all three of
doc_qa's defences have run, so an unverified claim can re-enter at the last
step, which is the exact failure the capability exists to prevent.

So `run` routes: a bare document question takes the fast path straight to
doc_qa (identical output, identical guarantees, one LLM call), and anything
that needs more than that goes to the agent, which still has swarn_doc_ask
in its toolset.

The routing decision is deliberately conservative in one direction. Guessing
"agent" for a question that could have been fast-pathed costs latency and
prints an extra paraphrase — annoying. Guessing "ask" for a task that needed
to train a model or write a file drops most of the requested work on the
floor — broken. So this module only claims the fast path when the task looks
like a question AND carries no sign of any other work, and the caller can
always force either way (`--ask` / `--agent`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

# Extensions the document stack can actually read. Anything else that appears
# on the command line is a path for the agent to deal with, not a document.
DOCUMENT_SUFFIXES = {
    ".pdf",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
}

# Verbs and nouns that mean the task needs tools the fast path does not have.
# Matched as whole words against the lowercased task. Kept deliberately short:
# every entry here is a capability `ask` genuinely cannot perform, not merely a
# word that sounds technical.
_OTHER_WORK = {
    # code / filesystem
    "write", "create", "edit", "refactor", "implement", "script", "install",
    "save", "export", "download", "delete", "rename", "commit",
    # ML pipeline
    "train", "model", "predict", "forecast", "classify", "regression",
    "hyperparameter", "tune", "fine-tune", "finetune", "benchmark",
    # analysis that needs execution
    "plot", "chart", "graph", "visualise", "visualize", "simulate",
    # platform verbs
    "index", "deploy", "serve", "package", "orchestrate",
}

# Phrasings that mean "answer from the document". Their presence is not
# required — a task with a document and no other-work signal is treated as a
# question anyway — but they let an imperative like "summarise page 3" take
# the fast path instead of being read as ambiguous.
_QUESTION_WORDS = {
    "what", "who", "when", "where", "which", "why", "how", "whose", "whom",
    "is", "are", "was", "were", "does", "do", "did", "can", "should",
    "list", "summarise", "summarize", "extract", "find", "tell", "show",
    "explain", "describe", "compare", "total", "how much", "how many",
}

_WORD = re.compile(r"[a-z][a-z\-]*")

# A bare token in the task text is only treated as a path if it carries a
# document suffix — "the invoice" must not be probed against the filesystem,
# but "invoice.pdf" should be.
#
# Deliberately no spaces inside the match. A path *can* contain them, but a
# pattern permitting that has no way to tell "total in invoice.pdf" (one word
# of path) from "my report file.pdf" (two), and the greedy reading swallows
# the whole sentence. A path with spaces belongs in the explicit trailing
# argument, where the shell has already delimited it for us.
_PATH_TOKEN = re.compile(
    r"""(?:^|[\s\"'\(\[])                       # start, space, quote or bracket
        ([\w\-./~\\:]+\.(?:pdf|png|jpe?g|tiff?|bmp|webp))
        (?=$|[\s\"'\)\],.;:])                    # ends at a boundary
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass
class Route:
    """Where `swarn run` should send this task, and why."""

    kind: str                                  # "ask" | "agent"
    documents: List[str] = field(default_factory=list)
    reason: str = ""

    @property
    def is_fast_path(self) -> bool:
        return self.kind == "ask"


def is_document(path: str | Path) -> bool:
    return Path(path).suffix.lower() in DOCUMENT_SUFFIXES


def find_document_paths(
    task: str,
    explicit: Optional[Sequence[str]] = None,
    cwd: Optional[str | Path] = None,
) -> List[str]:
    """
    Every readable document this invocation refers to, in the order named.

    Explicit trailing arguments come first (`swarn run "q" a.pdf b.pdf`), then
    any document-suffixed token found inside the task text itself — so
    `swarn run "what is the total in invoice.pdf"` works without the user
    having to repeat the path as an argument.

    A token is only returned if it exists on disk. A name that looks like a
    path but is not there is left for the agent to explain, rather than being
    fast-pathed into a confusing "no such file" from the document reader.
    """
    base = Path(cwd) if cwd else Path.cwd()
    found: List[str] = []

    def add(candidate: str) -> None:
        p = Path(candidate).expanduser()
        if not p.is_absolute():
            p = base / p
        resolved = str(p)
        if is_document(p) and p.is_file() and resolved not in found:
            found.append(resolved)

    for item in explicit or ():
        add(item)
    for match in _PATH_TOKEN.finditer(task):
        add(match.group(1).strip())
    return found


def mentions_other_work(task: str) -> bool:
    """True when the task asks for something the document reader cannot do."""
    words = set(_WORD.findall(task.lower()))
    return bool(words & _OTHER_WORK)


def looks_like_question(task: str) -> bool:
    """True when the task reads as a question about content."""
    lowered = task.lower().strip()
    if lowered.endswith("?"):
        return True
    words = _WORD.findall(lowered)
    if not words:
        return False
    # Either it opens with a question/request word, or one appears early
    # enough to be the main verb rather than an incidental mention.
    return bool(set(words[:3]) & _QUESTION_WORDS)


def route(
    task: str,
    explicit_paths: Optional[Sequence[str]] = None,
    cwd: Optional[str | Path] = None,
    force: Optional[str] = None,
) -> Route:
    """
    Decide how `swarn run` should handle this invocation.

    `force` is "ask" or "agent" for the CLI's override flags; it still resolves
    the document list, because `--ask` needs to know which file to read.
    """
    documents = find_document_paths(task, explicit_paths, cwd)

    if force == "ask":
        return Route("ask", documents, "forced with --ask")
    if force == "agent":
        return Route("agent", documents, "forced with --agent")

    if not documents:
        return Route("agent", documents, "no readable document named")
    if len(documents) > 1:
        # doc_qa answers about one document at a time; several of them is a
        # cross-document task, which is exactly what the agent loop is for.
        return Route("agent", documents,
                     f"{len(documents)} documents named — needs the agent to combine them")
    if mentions_other_work(task):
        return Route("agent", documents,
                     "task asks for work beyond reading the document")
    if not looks_like_question(task):
        return Route("agent", documents,
                     "task does not read as a question about the document")

    return Route("ask", documents, "a single question about a single document")
