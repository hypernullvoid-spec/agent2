"""
Grounded Document Question Answering
===================================

    swarn ask "what was the percentage increase in revenue" report.pdf

Answers a natural-language question about a document, and — the part that
matters — shows its work: the figures the answer was derived from, the page
and bounding box each one was read at, the arithmetic that turned them into
the answer, and an annotated image with the evidence boxed.

Why the evidence, and not just the answer
──────────────────────────────────────────
"What was the percentage increase in revenue" has a property worth being
explicit about: **the answer is not printed anywhere in the document.** No page
says "23.4%". Two figures are on the page and the percentage is derived from
them. So the honest unit of output here is not a number — it is a number plus
the two numbers it came from plus the operation, because that is the only form
a reviewer can actually check.

This matters more than usual because the derivation runs through an LLM, which
will produce a fluent, confident, plausible number whether or not the document
supports one. Three defences, in order of how much they actually protect you:

  1. The model never sees the document — it sees a TRANSCRIPT of words this
     repo extracted, with a stable id per line. It can only cite lines that
     exist.
  2. Every quote it returns is checked back against the line it claims to come
     from. A quote that is not on that line is dropped, not reported. This is
     the check that catches a fabricated citation, and it is done here in code
     rather than asked for in the prompt.
  3. The arithmetic is re-evaluated locally. If the model says
     "(148200-120100)/120100 = 31%", the mismatch is surfaced rather than
     printed as fact.

None of that makes the ANSWER guaranteed correct. It makes the answer
*checkable*, which is the difference between a tool you can put in front of a
reviewer and one you cannot.

Relationship to doc_intelligence
─────────────────────────────────
This module owns no extraction and no rendering. It sits on
`DocumentInspector.page_words()` for text (PDF text layer or OCR, chosen by the
same `auto` rules, never mock) and on `draw_bounding_boxes()` for the overlay.
The evidence boxes it draws are the same real coordinates `swarn doc-inspect`
would show for those words.
"""

from __future__ import annotations

import ast
import json
import operator
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from swarn.capabilities.doc_intelligence import (
    BoundingBox,
    DocumentInspector,
    DocumentIntelligenceError,
    ExtractedField,
    _box_of,
)

if TYPE_CHECKING:                       # pragma: no cover
    from swarn.capabilities.doc_store import StoredDocument, StoredPage

# How much document text to put in front of the model. Comfortably inside the
# context of any deployed endpoint while leaving room for the answer; a longer
# document is narrowed to its most relevant pages rather than truncated
# mid-sentence, so the model never sees half a table.
MAX_TRANSCRIPT_CHARS = 60_000

# Confidence assigned to a verified evidence span when drawing it. Evidence is
# not a probabilistic extraction — the quote either was found on the cited line
# or it was not — so verified spans render as high-confidence (green) and
# unverified ones as low (red), which is exactly what a reviewer should see.
VERIFIED_CONFIDENCE   = 0.95
UNVERIFIED_CONFIDENCE = 0.30

# Relative tolerance when re-checking the model's arithmetic. Loose enough for
# honest rounding in a stated result ("23.4%" for 23.397%), tight enough that a
# genuinely wrong computation fails.
COMPUTATION_TOLERANCE = 0.01

_LINE_ID_RE = re.compile(r"p(\d+):L(\d+)")


class DocumentQAError(DocumentIntelligenceError):
    """Question-answering failures (no LLM configured, unusable model reply)."""


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════


class EvidenceSpan(BaseModel):
    """One piece of the document the answer rests on."""

    model_config = ConfigDict(extra="ignore")

    page_number: int = Field(ge=1)
    line_id: str
    label: str                      # what the model says this figure is
    quote: str                      # the text as the model cited it
    box: BoundingBox
    verified: bool                  # was `quote` actually found in the document?
    source_line: str = ""           # the text as this repo actually read it
    # Which resolution strategy located it: line | multiline | table | none.
    # Worth surfacing — "found by unioning two table cells" is a materially
    # different claim from "found verbatim on the cited line", and a reviewer
    # auditing an answer should be able to tell them apart.
    strategy: str = "line"
    # Table provenance when strategy == "table": which table and row, which
    # columns were matched, and each matched cell's own text and box.
    table: Optional[dict] = None

    def as_field(self) -> ExtractedField:
        """Render as an ExtractedField so the existing overlay renderer can
        draw it — evidence and extracted fields are the same thing on a page."""
        return ExtractedField(
            field_name=self.label or "evidence",
            field_value=self.quote,
            confidence=VERIFIED_CONFIDENCE if self.verified else UNVERIFIED_CONFIDENCE,
            box=self.box,
        )


class DocumentAnswer(BaseModel):
    """The answer plus everything needed to check it."""

    model_config = ConfigDict(extra="ignore")

    question: str
    document_name: str
    answer: str
    found: bool = True
    computation: str = ""
    computation_check: str = ""
    evidence: List[EvidenceSpan] = Field(default_factory=list)
    annotated_image_paths: List[str] = Field(default_factory=list)
    pages_searched: List[int] = Field(default_factory=list)
    page_count: int = 0
    backend: str = ""
    document_id: str = ""
    # True when this answer was produced from the persisted structured
    # document rather than a fresh parse — i.e. the PDF was not re-read.
    from_store: bool = False
    raw_json: dict = Field(default_factory=dict)

    @property
    def unverified(self) -> List[EvidenceSpan]:
        return [span for span in self.evidence if not span.verified]

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def summary(self) -> str:
        """Terminal digest: the answer, how it was derived, and where from."""
        lines = [self.answer if self.found else f"Not answerable from this document. {self.answer}"]

        if self.computation:
            check = f"  [{self.computation_check}]" if self.computation_check else ""
            lines += ["", f"  computed  {self.computation}{check}"]

        if self.evidence:
            lines += ["", "  evidence"]
            width = max(len(span.label) for span in self.evidence)
            for span in self.evidence:
                # "on that line" was accurate when the line was the only place
                # searched. Resolution now covers the cited line, the lines
                # after it, and the tables on every page — so an unverified
                # span really means "nowhere in this document", and saying so
                # is the difference between a warning a reviewer trusts and
                # one they learn to ignore.
                mark = " " if span.verified else "  << NOT FOUND IN DOCUMENT"
                box = span.box
                lines.append(
                    f"    p{span.page_number}  {span.label:<{width}}  {span.quote:<28}"
                    f"  box({box.xmin:.3f}, {box.ymin:.3f}, {box.xmax:.3f}, {box.ymax:.3f})"
                    f"{mark}")

        for path in self.annotated_image_paths:
            lines += ["", f"  -> {path}"]
        if self.document_id:
            origin = "stored" if self.from_store else "ingested now"
            lines += ["", f"  [{origin}: {self.document_id} · {self.backend} · "
                          f"{len(self.pages_searched)}/{self.page_count} pages searched]"]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# TRANSCRIPT — what the model is allowed to see
# ══════════════════════════════════════════════════════════════════════════════


class PageTranscript:
    """
    One page as numbered lines, for the model to read and for citations to
    resolve against.

    Now a thin view over a StoredPage rather than a grouping step of its own.
    Lines are built once, at ingest (doc_store._lines_for_page), which is what
    makes `line_id` durable: a citation of "p3:L5" recorded today has to mean
    the same line tomorrow, and it cannot if the grouping is recomputed per
    question from code that might have changed in between.
    """

    def __init__(self, page: "StoredPage"):
        self.page = page
        self.page_number = page.page_number
        self.page_size = page.page_size

    @property
    def lines(self) -> List[dict]:
        return [{"line_id": line.line_id, "text": line.text,
                 "column": line.column, "words": line.word_dicts()}
                for line in self.page.lines]

    def render(self) -> str:
        return self.page.render()

    def find(self, line_id: str) -> Optional[dict]:
        return self.page.find(line_id)


def build_transcript(
    inspector: DocumentInspector,
    file_path: str,
    pages: Sequence[int],
    backend: Optional[str] = None,
    document: Optional["StoredDocument"] = None,
    artifacts_dir: Optional[Union[str, Path]] = None,
) -> Tuple[dict, str]:
    """
    Transcribe the given pages. Returns ({page_number: PageTranscript}, backend).

    Reads the persisted structured document, ingesting it once if this file has
    not been seen before. The PDF itself is opened for text exactly once per
    distinct file CONTENT, no matter how many questions get asked about it.
    """
    from swarn.capabilities.doc_store import get_or_ingest

    if document is None:
        # Same scoping rule as ask_document: an inspector pointed at a private
        # artifacts directory keeps its documents there too, so a caller that
        # isolates one never writes into the shared store.
        document, _ = get_or_ingest(
            file_path, inspector=inspector, backend=backend,
            artifacts_dir=artifacts_dir or inspector.artifacts_dir)

    transcripts: dict = {}
    for page_number in pages:
        page = document.page(page_number)
        if page is not None:
            transcripts[page_number] = PageTranscript(page)
    return transcripts, document.backend


def select_pages(transcripts: dict, question: str, budget: int = MAX_TRANSCRIPT_CHARS) -> List[int]:
    """
    Choose which pages to send when the whole document will not fit.

    Ranked by how many of the question's terms a page contains, because the
    page holding the revenue table is the one that says "revenue". Truncating
    the transcript instead would cut a table in half and invite exactly the
    kind of half-informed answer this module is trying to prevent.
    """
    rendered = {page: t.render() for page, t in transcripts.items()}
    if sum(len(text) for text in rendered.values()) <= budget:
        return sorted(rendered)

    terms = {term for term in re.findall(r"[a-z0-9]{3,}", question.lower())}
    scored = sorted(
        rendered,
        key=lambda page: (
            -sum(rendered[page].lower().count(term) for term in terms),
            page,
        ),
    )
    kept, used = [], 0
    for page in scored:
        size = len(rendered[page]) + 1
        if used + size > budget and kept:
            break
        kept.append(page)
        used += size
    return sorted(kept)


# ══════════════════════════════════════════════════════════════════════════════
# ARITHMETIC VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

_SAFE_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
}
_SAFE_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_eval(expression: str) -> float:
    """
    Evaluate a pure-arithmetic expression.

    Deliberately not `eval`: this string comes from an LLM, and an LLM's output
    is untrusted input no matter how arithmetic-shaped it looks. Only numeric
    literals and the operators above survive the walk — a name, call, attribute,
    or subscript raises.
    """
    tree = ast.parse(expression.strip(), mode="eval")

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(f"non-numeric constant: {node.value!r}")
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
            return _SAFE_BINOPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARYOPS:
            return _SAFE_UNARYOPS[type(node.op)](walk(node.operand))
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    return walk(tree)


def verify_computation(computation: str) -> str:
    """
    Re-run the model's arithmetic and report whether it holds.

    Returns "verified", "MISMATCH: ...", or "" when there is nothing checkable.
    An unverifiable computation is not an error — plenty of correct answers are
    a lookup with no arithmetic at all — but a WRONG one must never print as
    though it were fact.

    A compound answer ("margin FY24 = ...; margin FY23 = ...; difference =
    ...") is split on ';' and each part checked, because a question like "did
    the margin improve" genuinely needs three steps and checking none of them
    would waste the check exactly where the arithmetic is most error-prone.
    """
    parts = [part for part in str(computation).split(";") if part.strip()]
    if len(parts) > 1:
        results = [_verify_one(part) for part in parts]
        problems = [r for r in results if r.startswith("MISMATCH")]
        if problems:
            return "; ".join(problems)
        return "verified" if any(r == "verified" for r in results) else ""
    return _verify_one(computation)


def _verify_one(computation: str) -> str:
    """
    Check a single claim of the form `expression = result`.

    Chained equalities are handled, because models write them constantly:
    "(a - b) / b = 0.23397 = 23.4%" states the raw ratio and its percentage
    form. The claim holds if the expression matches ANY stated value — they
    are the same quantity in different units, and demanding it match the last
    one would fail a correct answer for choosing a different final unit.
    """
    if not computation or "=" not in computation:
        return ""

    segments = [segment.strip() for segment in computation.split("=")]
    expression = segments[0].replace(",", "").replace("%", "")
    claims = [segment for segment in segments[1:] if segment]
    if not claims:
        return ""

    try:
        actual = safe_eval(expression)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError):
        return ""

    checked: list = []
    for claim in claims:
        cleaned = claim.replace(",", "")
        # The stated result often trails an aside — "0.3097 (30.97%)",
        # "23.4% YoY". Compare against the first number in it.
        number = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if not number:
            continue
        expected = float(number.group(0))
        percent = "%" in cleaned and cleaned.index(number.group(0)) < cleaned.find("%")

        # Both of these are idiomatic ways to write the same percentage:
        #     (a - b) / b       = 23.4%      <- ratio on the left
        #     (a - b) / b * 100 = 23.4%      <- already scaled on the left
        # Guessing which one the model meant produces false alarms: an earlier
        # version assumed the first, so the second came back as "2,340%, not
        # 23.4%" — crying wolf on correct arithmetic. Accepting either reading
        # costs nothing that matters, because a genuinely wrong computation
        # (claiming 31% for these figures) fails under both.
        candidates = [actual, actual * 100] if percent else [actual]
        if any(abs(value - expected) / max(abs(expected), abs(value), 1e-9)
               <= COMPUTATION_TOLERANCE for value in candidates):
            return "verified"
        checked.append((expected, claim, "%" if percent else ""))

    if not checked:
        return ""
    _, claim, suffix = checked[-1]
    return f"MISMATCH: {expression} = {actual:,.6g}{suffix}, not {claim}"


# ══════════════════════════════════════════════════════════════════════════════
# THE MODEL CALL
# ══════════════════════════════════════════════════════════════════════════════

# Budget for the reply. Generous because reasoning-style models spend tokens
# thinking before they answer, and a reply cut off mid-JSON is unparseable —
# which reads to the caller as "the model failed" when it merely ran out of
# room. Cheap insurance against a confusing failure mode.
ANSWER_MAX_TOKENS = 4000

_SYSTEM_PROMPT = (
    "You answer questions about a document using ONLY the transcript you are "
    "given. You reply with ONE JSON object and nothing else: no reasoning, no "
    "explanation before or after it, no markdown fences. Begin your reply with "
    "the '{' character. Do all of your reasoning silently and put the result "
    "in the JSON fields."
)

_USER_TEMPLATE = """\
Below is a transcript of a document. Each line is prefixed with a stable id
like [p3:L12] meaning page 3, line 12.

TRANSCRIPT
----------
{transcript}
----------

QUESTION: {question}

Answer with one JSON object of exactly this shape:

{{"found": true|false,
  "answer": "<a direct, one-or-two-sentence answer>",
  "computation": "<the arithmetic, e.g. (148200 - 120100) / 120100 = 23.4% — \
empty string if the answer needed no arithmetic>",
  "evidence": [{{"line_id": "<id from the transcript>",
                 "label": "<what this figure is, e.g. FY24 revenue>",
                 "quote": "<the exact text from that line that you used>"}}]}}

Rules — these are checked automatically after you answer:
- Use ONLY the transcript. If it does not contain what is needed, set
  "found": false and say plainly what is missing. Do not guess, do not use
  outside knowledge, and do not fill a gap with a typical or expected value.
- Every "line_id" must be one that appears above.
- Every "quote" must be text copied EXACTLY from that line, and must be the
  SHORTEST piece of it that carries the figure you used — usually just the
  number itself ("120100"), NOT the whole line. The quote decides which words
  get boxed on the page, so a whole-line quote boxes the whole line and points
  at nothing in particular. A `|` means TABLE CELLS in one row
  (`Label | cell | cell`) — quote the one cell, never the whole row.
- Lines are in reading order. On a multi-column page the transcript gives one
  column in full before starting the next, so a caption applies to the lines
  BELOW IT, not to a line that merely sits at the same height elsewhere on the
  page. Never pair a label with a value unless they are adjacent here.
- Cite every figure the answer depends on, one evidence entry each — for a
  change, a ratio, or a margin, that means every number that went into the
  arithmetic, including ones from different rows.
- "computation" is re-evaluated. Write it as a plain arithmetic expression
  equal to your result, using the raw numbers (no thousands separators).

Reply with the JSON object only. Start at '{{' and stop at the closing '}}'.
"""


def _default_client():
    from agent.llm import create_client
    return create_client()


def ask_model(client, transcript: str, question: str) -> dict:
    """Send the transcript and question; return the parsed JSON object."""
    from swarn.capabilities.doc_intelligence import parse_vlm_response

    prompt = _USER_TEMPLATE.format(transcript=transcript, question=question)
    try:
        # temperature=0: the same document and question should give the same
        # answer twice. A grounded-extraction tool that varies run to run is
        # not one anybody can build a review process around.
        reply = client.complete(_SYSTEM_PROMPT, prompt, temperature=0,
                                max_tokens=ANSWER_MAX_TOKENS)
    except Exception as exc:                                           # noqa: BLE001
        raise DocumentQAError(
            f"the question could not be answered because the LLM call failed: {exc}\n"
            "The deployed endpoint is configured in agent/llm/router.py "
            "(SWARN_DEPLOYED_BASE_URL / SWARN_DEPLOYED_API_KEY)."
        ) from exc

    # parse_vlm_response handles fenced and prose-wrapped JSON — the same
    # unreliability applies whether the model is looking at pixels or text.
    try:
        return parse_vlm_response(reply)
    except DocumentIntelligenceError as exc:
        # Distinguish "the model misbehaved" from "your document is bad", and
        # name the usual cause: a reasoning model that narrated its way past
        # the token budget before emitting any JSON.
        raise DocumentQAError(
            f"the model did not return usable JSON ({exc}).\n"
            "This usually means it answered in prose instead, or ran out of tokens "
            "while reasoning. Try a more capable model via SWARN_DEPLOYED_MODEL, or "
            "ask a narrower question (--page N) so the reply has more room."
        ) from exc


# ══════════════════════════════════════════════════════════════════════════════
# EVIDENCE RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════


def _normalize(text: str) -> str:
    """
    Canonical form for comparing a quote against the page's words.

    Drops whitespace, thousands separators, and the "|" column marker. That
    last one is not cosmetic: the transcript renders a table row as
    `Revenue | 120100 | 148200`, and models routinely quote the row back
    verbatim, pipes and all. Those pipes were added by the renderer and appear
    in no word on the page, so without stripping them here a perfectly correct
    citation fails verification and a right answer gets reported as
    unsupported — which is a worse failure than the one verification exists to
    prevent.
    """
    return re.sub(r"[\s,|]", "", str(text)).lower()


# How many consecutive lines a single quote may span before the multi-line
# strategy gives up. Wrapped prose and a wrapped table cell both stay well
# inside this; a larger window would let a quote "match" by sweeping up
# unrelated neighbouring text.
MAX_EVIDENCE_LINE_SPAN = 4

# A table match must consume at least this fraction of the quote's tokens from
# actual CELL text. Column headers are matched too — models write
# "<trigger> CC 5", where "CC" is the column label, not a value — but a match
# carried mostly by header words would let any string built from column names
# validate against any row.
MIN_CELL_TOKEN_SHARE = 0.5

# Below this many tokens a quote is too short to identify a table row: "5"
# matches a numeric cell in almost any table. Short quotes stay on the line
# strategies, where the cited line pins them down.
MIN_TABLE_QUOTE_TOKENS = 2


def _tokens(text: str) -> List[str]:
    """Normalized word tokens: lowercase, punctuation-free.

    This is the comparison currency for table matching, and it absorbs exactly
    the differences that are not meaningful — case, whitespace, line breaks
    inside a wrapped cell, hyphens and brackets from the source styling, and
    the "|" the transcript adds between columns. It does NOT absorb word
    identity: every token still has to be there, in order.
    """
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split()


def _token_span(haystack: Sequence[str], needle: Sequence[str]) -> int:
    """Index where `needle` occurs as a contiguous run in `haystack`, else -1."""
    if not needle or len(needle) > len(haystack):
        return -1
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start:start + len(needle)] == list(needle):
            return start
    return -1


def resolve_evidence(
    raw_evidence: Any, transcripts: dict, document: Optional["StoredDocument"] = None,
) -> List[EvidenceSpan]:
    """
    Turn the model's citations into boxed, verified evidence.

    Strategies are tried in order of how precisely they pin the quote down, and
    the first that succeeds wins:

      1. line       the quote is on the cited line (the original behaviour, and
                    still the fast path for ordinary prose)
      2. multiline  the quote spans the cited line and the ones after it —
                    wrapped prose, or a cell whose text runs over several lines
      3. table      the quote is composed of table CELLS, possibly across
                    columns, plus the column headers a model naturally writes
                    between them

    Strategy 3 exists because a line-centric resolver is wrong about tables.
    A model asked to cite a row writes what it reads across the row —
    "<trigger> CC 5" — and that string is on no single extracted line: it is
    one cell, then a column header, then a cell in a different column. The old
    resolver reported that correct citation as NOT FOUND, which is the worst
    kind of failure here, because it teaches a reviewer to distrust a warning
    that is usually right.

    Widening what counts as found does NOT weaken verification: every strategy
    still requires the quote's tokens to be physically present, in order, in
    the words it claims to come from. What changes is only the unit those words
    are allowed to span.
    """
    if not isinstance(raw_evidence, list):
        return []

    spans: List[EvidenceSpan] = []
    for item in raw_evidence:
        if not isinstance(item, dict):
            continue
        # The transcript displays ids as "[p3:L12] text", so models cite them
        # bracketed about as often as bare. Be liberal about the FORM of the
        # reference and strict about what it points AT: a citation is rejected
        # for naming a line that does not exist, never for punctuation.
        match = _LINE_ID_RE.search(str(item.get("line_id") or ""))
        if not match:
            continue
        page_number = int(match.group(1))
        line_id = f"p{match.group(1)}:L{match.group(2)}"
        transcript = transcripts.get(page_number)
        if transcript is None:
            continue
        line = transcript.find(line_id)
        if line is None:
            continue

        quote = " ".join(str(item.get("quote") or "").split())
        label = " ".join(str(item.get("label") or "").split()) or "evidence"

        span = (_resolve_on_line(quote, line, transcript, page_number, line_id, label)
                or _resolve_across_lines(quote, line_id, transcript, page_number, label)
                or _resolve_in_tables(quote, page_number, label, transcripts, document))
        if span is None:
            # Nothing located it. Report the citation as unverified against the
            # line it named — the reviewer needs to see WHAT was claimed and
            # where the model thought it was.
            span = EvidenceSpan(
                page_number=page_number, line_id=line_id, label=label,
                quote=quote or line["text"],
                box=_box_of(line["words"], *transcript.page_size),
                verified=False, source_line=line["text"], strategy="none",
            )
        spans.append(span)
    return spans


def _resolve_on_line(
    quote: str, line: dict, transcript, page_number: int, line_id: str, label: str,
) -> Optional[EvidenceSpan]:
    """Strategy 1 — the quote sits on the cited line. Unchanged fast path."""
    matched = _words_matching(line["words"], quote)
    if not matched:
        return None
    return EvidenceSpan(
        page_number=page_number, line_id=line_id, label=label,
        quote=quote or line["text"],
        box=_box_of(matched, *transcript.page_size),
        verified=True, source_line=line["text"], strategy="line",
    )


def _resolve_across_lines(
    quote: str, line_id: str, transcript, page_number: int, label: str,
) -> Optional[EvidenceSpan]:
    """
    Strategy 2 — the quote runs from the cited line into the ones below it.

    Covers wrapped prose and a wrapped table cell alike. The words of the
    window are concatenated and handed to the SAME shortest-run matcher the
    line strategy uses, so the resulting box is still tight around the quote
    rather than around every line it touched.
    """
    lines = transcript.lines
    start = next((i for i, line in enumerate(lines) if line["line_id"] == line_id), None)
    if start is None:
        return None

    for end in range(start + 1, min(start + MAX_EVIDENCE_LINE_SPAN, len(lines)) + 1):
        window = lines[start:end]
        # Never run the window past a column change. On a multi-column page the
        # next line can be at the opposite side of the sheet, and a match
        # spanning that boundary would box most of the page — pointing at
        # everything, which is the same as pointing at nothing.
        if any(line.get("column") != window[0].get("column") for line in window):
            break
        words = [word for line in window for word in line["words"]]
        matched = _words_matching(words, quote)
        if matched:
            return EvidenceSpan(
                page_number=page_number, line_id=line_id, label=label, quote=quote,
                box=_box_of(matched, *transcript.page_size),
                verified=True,
                source_line=" ".join(line["text"] for line in window),
                strategy="multiline",
            )
    return None


def _resolve_in_tables(
    quote: str, cited_page: int, label: str, transcripts: dict,
    document: Optional["StoredDocument"],
) -> Optional[EvidenceSpan]:
    """
    Strategy 3 — the quote is assembled from table cells.

    The cited page is searched first, then the rest of the document: a model
    that reads a row correctly may still misremember which page it was on, and
    the row itself is the evidence. When a match is found elsewhere the span
    carries the page it was ACTUALLY found on, so the citation a reviewer sees
    points at the right place.
    """
    tokens = _tokens(quote)
    if len(tokens) < MIN_TABLE_QUOTE_TOKENS:
        return None

    pages = sorted(transcripts, key=lambda page: (page != cited_page, page))
    for page_number in pages:
        page = transcripts[page_number].page
        for table in page.tables:
            match = _match_table(tokens, table)
            if match is None:
                continue
            cells = match["cells"]
            box = _union_box(cells, transcripts[page_number].page_size)
            return EvidenceSpan(
                page_number=page_number, line_id=f"p{page_number}:T{table.index}",
                label=label, quote=quote, box=box, verified=True,
                source_line=" | ".join(cell.text for cell in match["row"].cells if cell.text),
                strategy="table",
                table={
                    "table_index": table.index,
                    "row_index": match["row"].index,
                    "columns": [cell.column for cell in cells],
                    "cells": [{"column": cell.column, "text": cell.text, "bbox": cell.bbox}
                              for cell in cells],
                    "page_corrected": page_number != cited_page,
                },
            )
    return None


def _match_table(tokens: Sequence[str], table) -> Optional[dict]:
    """The row whose cells (and column headers) compose this quote, if any."""
    header_tokens = [_tokens(header) for header in table.headers]
    for row in table.rows:
        cells = _consume_row(tokens, row, header_tokens)
        if cells:
            return {"row": row, "cells": cells}
    return None


def _consume_row(tokens: Sequence[str], row, header_tokens: Sequence[Sequence[str]]):
    """
    Walk the quote left to right, spending its tokens against this row.

    At each position the longest available match wins, from three sources:

      * a cell, matched from its start — models quote a cell's opening words
        and stop, so "Cross-Functional Team Bonus" must match the cell
        "Cross-Functional Team Bonus (3+ departments)"
      * a column header — the label a model writes between two values
      * a cell containing the whole remainder — a quote lifted from the middle
        of one long cell

    Returns the matched CELLS (headers contribute no box, being labels rather
    than data) or None if any token cannot be spent. Requiring every token to
    be consumed is what keeps this from being fuzzy matching: an invented
    quote runs out of ways to spend its words almost immediately.
    """
    remaining = list(tokens)
    matched: List = []
    cell_tokens_used = 0

    while remaining:
        best_length = 0
        best_cell = None

        for cell in row.cells:
            cell_tokens = _tokens(cell.text)
            if not cell_tokens:
                continue
            # (a) the quote continues with this cell's opening words
            length = 0
            for index in range(min(len(remaining), len(cell_tokens))):
                if remaining[index] != cell_tokens[index]:
                    break
                length = index + 1
            if length > best_length:
                best_length, best_cell = length, cell
            # (b) the rest of the quote sits inside this cell
            if _token_span(cell_tokens, remaining) >= 0 and len(remaining) > best_length:
                best_length, best_cell = len(remaining), cell

        for header in header_tokens:
            if header and remaining[:len(header)] == list(header) and len(header) > best_length:
                best_length, best_cell = len(header), None

        if best_length == 0:
            return None
        if best_cell is not None:
            cell_tokens_used += best_length
            if best_cell not in matched:
                matched.append(best_cell)
        remaining = remaining[best_length:]

    if not matched or cell_tokens_used < len(tokens) * MIN_CELL_TOKEN_SHARE:
        return None
    return matched


def _union_box(cells: Sequence, page_size) -> BoundingBox:
    """One box covering every matched cell.

    For a quote spanning two columns this is deliberately the union rather than
    two separate boxes: the evidence is the ROW's statement, and a reviewer
    checking "<trigger> CC 5" needs to see the trigger and the value together.
    """
    boxes = [cell.bbox for cell in cells if cell.bbox and len(cell.bbox) == 4]
    if not boxes:
        return BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1)
    return BoundingBox.from_pixels(
        (min(b[0] for b in boxes), min(b[1] for b in boxes),
         max(b[2] for b in boxes), max(b[3] for b in boxes)),
        *page_size)


def _words_matching(words: Sequence[dict], quote: str) -> List[dict]:
    """
    The SHORTEST run of consecutive words whose text contains the quote.

    Shortest, not first: on a table row `Revenue | 120100 | 148200`, a scan
    that stops at its first hit would return "Revenue 120100" for the quote
    "120100" and box both cells. The box is the whole point here — it has to
    land on the figure being cited, not on the neighbourhood it lives in.

    Returns [] when the quote is not on the line at all, which is the signal
    that the model cited something it did not read. Checked in code rather
    than trusted from the prompt, because a prompt rule is a request and this
    is a guarantee.
    """
    target = _normalize(quote)
    if not target:
        return []

    best: Optional[List[dict]] = None
    for start in range(len(words)):
        joined = ""
        for end in range(start, len(words)):
            joined += _normalize(words[end]["text"])
            if target in joined:
                run = list(words[start:end + 1])
                if best is None or len(run) < len(best):
                    best = run
                break
            if len(joined) > len(target) * 3:
                break
    return best or []


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def ask_document(
    file_path: str,
    question: str,
    pages: Optional[Sequence[int]] = None,
    backend: Optional[str] = None,
    annotate: bool = True,
    artifacts_dir: Optional[Union[str, Path]] = None,
    client: Any = None,
    inspector: Optional[DocumentInspector] = None,
    on_ingest=None,
) -> DocumentAnswer:
    """
    Answer `question` about `file_path`, grounded in the document's own words.

    `client` is injectable so the pipeline can be tested end to end without a
    network call; left unset it uses the deployed endpoint from
    agent/llm/router.py, exactly like every other LLM call in this repo.
    """
    if not str(question).strip():
        raise DocumentQAError("the question is empty.")

    from swarn.capabilities.doc_store import get_or_ingest

    inspector = inspector or DocumentInspector(artifacts_dir=artifacts_dir)
    # The store lives under the inspector's artifacts root unless told
    # otherwise, so a caller that scopes the inspector to a temp directory
    # (tests, a request-scoped run) gets its documents scoped there too rather
    # than writing into the shared artifacts tree.
    artifacts_dir = artifacts_dir or inspector.artifacts_dir

    # Load the parsed document, ingesting it once if this content is new. Every
    # later question about the same file skips extraction entirely — which is
    # the difference between milliseconds and a full OCR pass per question.
    document, ingested_now = get_or_ingest(
        file_path, inspector=inspector, backend=backend,
        artifacts_dir=artifacts_dir, on_ingest=on_ingest)
    page_count = document.page_count

    wanted = list(pages) if pages else list(range(1, page_count + 1))
    for page_number in wanted:
        if not 1 <= page_number <= page_count:
            raise DocumentQAError(
                f"page {page_number} is out of range ({page_count} pages in "
                f"{Path(file_path).name}).")

    transcripts, chosen_backend = build_transcript(
        inspector, file_path, wanted, backend, document=document)
    searched = select_pages(transcripts, question)
    if not searched:
        raise DocumentQAError(
            f"no readable text found in {Path(file_path).name} — nothing to answer from.")
    transcript_text = "\n".join(transcripts[page].render() for page in searched)

    payload = ask_model(client or _default_client(), transcript_text, question)

    answer_text = " ".join(str(payload.get("answer") or "").split())
    found = bool(payload.get("found", True)) and bool(answer_text)
    computation = " ".join(str(payload.get("computation") or "").split())
    evidence = resolve_evidence(payload.get("evidence"), transcripts, document)

    # An answer asserted with no surviving evidence is exactly the shape of a
    # confident fabrication, so it is reported as unsupported rather than as an
    # answer. The model's text is kept — it usually explains what it was
    # reaching for — but `found` no longer claims the document backs it.
    if found and not any(span.verified for span in evidence):
        found = False
        answer_text = (answer_text + "  [unsupported: no cited text could be located "
                                     "in this document]").strip()

    annotated: List[str] = []
    if annotate and evidence:
        annotated = _annotate_evidence(inspector, file_path, evidence, document)

    return DocumentAnswer(
        question=question,
        document_name=Path(file_path).name,
        answer=answer_text or "The model returned no answer.",
        found=found,
        computation=computation,
        computation_check=verify_computation(computation),
        evidence=evidence,
        annotated_image_paths=annotated,
        pages_searched=searched,
        page_count=page_count,
        backend=chosen_backend,
        raw_json={
            "model_response": payload,
            "pages_transcribed": wanted,
            "n_evidence": len(evidence),
            "n_verified": sum(1 for span in evidence if span.verified),
            "document_id": document.document_id,
            "ingested_now": ingested_now,
            "ingested_at": document.ingested_at,
        },
        document_id=document.document_id,
        from_store=not ingested_now,
    )


def _annotate_evidence(
    inspector: DocumentInspector, file_path: str, evidence: Sequence[EvidenceSpan],
    document: Optional["StoredDocument"] = None,
) -> List[str]:
    """One annotated image per page that carries evidence."""
    by_page: dict = {}
    for span in evidence:
        by_page.setdefault(span.page_number, []).append(span)

    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(file_path).stem) or "document"
    paths = []
    for page_number, spans in sorted(by_page.items()):
        image = _page_image(inspector, file_path, page_number, document)
        paths.append(inspector.draw_bounding_boxes(
            image, [span.as_field() for span in spans],
            f"{stem}_p{page_number}_evidence.png"))
    return paths


def _page_image(
    inspector: DocumentInspector, file_path: str, page_number: int,
    document: Optional["StoredDocument"] = None,
):
    """
    The page picture to draw evidence on.

    Coordinates come from the store, but PIXELS cannot: an image is not
    recoverable from word geometry. `swarn ingest --render-pages` caches a
    raster per page for exactly this, and it is used when present; otherwise
    the source is rasterized on demand. Note that rasterizing is not text
    extraction — no pdfplumber pass, no OCR — so the "parse once" guarantee
    holds either way. `--no-annotate` skips this path entirely.
    """
    from PIL import Image

    page = document.page(page_number) if document else None
    if page is not None and page.image_path and Path(page.image_path).exists():
        with Image.open(page.image_path) as cached:
            return cached.convert("RGB")
    image, _ = inspector.load_page_image(file_path, page_number)
    return image


def answer_question(
    file_path: str,
    question: str,
    page: Optional[int] = None,
    backend: Optional[str] = None,
    annotate: bool = True,
    include_raw: bool = False,
) -> dict:
    """
    Registration surface for the agent tool (`swarn_doc_ask`) and the CLI:
    a plain JSON-serializable dict, never raising for an input problem.
    """
    try:
        result = ask_document(
            file_path, question,
            pages=[page] if page else None,
            backend=backend, annotate=annotate)
    except DocumentIntelligenceError as exc:
        return {"error": str(exc)}

    payload = result.to_dict()
    if not include_raw:
        payload.pop("raw_json", None)
    return payload


__all__ = [
    "DocumentAnswer",
    "DocumentQAError",
    "EvidenceSpan",
    "PageTranscript",
    "answer_question",
    "ask_document",
    "build_transcript",
    "resolve_evidence",
    "safe_eval",
    "select_pages",
    "verify_computation",
]
