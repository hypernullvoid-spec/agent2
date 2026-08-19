"""
Visual Document Intelligence & Bounding-Box Inspector
=====================================================

Turns a document (PDF or image) into *grounded* structured data: not just
"invoice_number = INV-2024-0917", but "invoice_number = INV-2024-0917, and
here is the rectangle on page 1 where I read it from" — plus an annotated
PNG that draws those rectangles so a human can audit the extraction in one
glance instead of re-reading the source document.

Why bounding boxes, and not just fields
────────────────────────────────────────
Phase 13's multimodal_rag.py already extracts a PDF's text, tables, and
key/value pairs (`extract_pdf_document`). What it cannot tell you is *where*
on the page a value came from, and that is exactly what turns an extraction
pipeline into something you can actually put in front of a reviewer. For
invoice/KYC/financial-statement work the failure mode is never "the model
returned nothing" — it is "the model returned a confident, plausible, wrong
number." A coordinate makes that falsifiable: the reviewer looks at the box,
sees the box is drawn around the *subtotal* row when the field claims to be
the *total*, and rejects it in a second. So every extracted field here
carries a `BoundingBox`, and every run can emit an annotated image.

Three interchangeable extraction backends
──────────────────────────────────────────
  vlm    An API-based vision model over an OpenAI-compatible /v1 endpoint
         (the same wire format OpenAI, Gemini, and Qwen2.5-VL servers all
         speak). The image goes up as a base64 data URL and the model is
         asked for JSON with `box_2d` in the standard normalized
         [ymin, xmin, ymax, xmax] 0–1000 convention. Selected only when a
         VLM API key is explicitly configured.
  ocr    Fully local: pytesseract word boxes, grouped into lines, mined for
         "Label: value" pairs. Coordinates are real (they come from the OCR
         engine, not from a language model's guess), and the label/value
         parsing reuses multimodal_rag.MultiModalIndexer._kv_match so a
         colon inside ordinary prose does not manufacture a field. Needs the
         tesseract binary, so it is opt-in.
  mock   A deterministic, realistic sample invoice extraction. This is the
         default when nothing else is configured, and it is the reason this
         module can be demoed and unit-tested out of the box with no API
         key, no GPU, and no network. The mock's field boxes and the mock
         invoice renderer (`render_mock_invoice`) are generated from the
         SAME layout table (`MOCK_INVOICE_LAYOUT`), so the boxes genuinely
         line up with the rendered document instead of being decorative.

Everything degrades to an error *string* or a mock result rather than an
exception at import time, matching how agent/tools.py reports optional-
dependency problems elsewhere in this codebase.

Entry points
─────────────
  DocumentInspector.process_document(path)   → DocumentExtractionResult
  inspect_document(path)                     → plain JSON-able dict
                                               (this is what agent/tools.py
                                               registers as swarn_doc_inspect)
  render_mock_invoice()                       → a synthetic PIL invoice image
  swarn doc-inspect <path>                     → CLI front end (agent/cli.py)
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ─── paths ────────────────────────────────────────────────────────────────────

# Repo root is three levels up: swarn/capabilities/doc_intelligence.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Annotated images land here. Phase 9 writes its figures to workspace/plots/,
# but these are *review artifacts* for a human rather than agent working
# files, and the demo script's contract names artifacts/ explicitly — so this
# capability gets its own top-level directory, overridable for callers that
# want the output somewhere else (a request-scoped temp dir, say).
ARTIFACTS_DIR = Path(os.environ.get("SWARN_ARTIFACTS_DIR", _REPO_ROOT / "artifacts"))

# ─── extraction tuning ────────────────────────────────────────────────────────

# Confidence tiers. Chosen so the three bands mean genuinely different things
# to a reviewer: HIGH = ship it, MEDIUM = glance at the box, LOW = read the
# source document. Anything finer-grained just becomes noise on the overlay.
CONFIDENCE_HIGH   = 0.85
CONFIDENCE_MEDIUM = 0.60

# Box colours, high-contrast against white/off-white document backgrounds.
TIER_COLORS: dict[str, tuple[int, int, int]] = {
    "high":   (22, 163, 74),    # green
    "medium": (217, 119, 6),    # amber
    "low":    (220, 38, 38),    # red
}

BOX_FILL_ALPHA  = 34     # translucent wash inside the box — must never obscure the text under it
TAG_FILL_ALPHA  = 235    # label tag is near-opaque: it has to stay readable over dense print
LABEL_MAX_CHARS = 38     # longer values are ellipsized on the overlay (never in the JSON)

DEFAULT_PDF_DPI = 150    # enough detail for OCR/VLM without producing 20 MB page rasters

# A coordinate above this is read as the 0–1000 integer convention rather than
# a 0.0–1.0 float. The threshold is 2.0, not 1.0, so a model that returns
# 1.02 from rounding noise gets clamped to 1.0 instead of collapsing to 0.001.
_THOUSANDTHS_THRESHOLD = 2.0

# Extraction backends.
#   vlm   API vision model (real coordinates, from the model)
#   text  the PDF's own embedded text layer (real coordinates, exact text)
#   ocr   rasterize + tesseract (real coordinates, transcribed text)
#   mock  SYNTHETIC sample data — see _resolve_backend for why `auto` must
#         never choose it, and MOCK_ONLY_ON_REQUEST below.
BACKENDS = ("vlm", "text", "ocr", "mock")

# Backends that read the document the caller actually supplied. Anything not
# in here is producing data from somewhere other than the document, which is
# only ever legitimate when it was explicitly requested.
REAL_BACKENDS = ("vlm", "text", "ocr")

# Minimum characters on a page before its embedded text layer is considered
# usable. A scanned PDF frequently carries a handful of stray characters (a
# stamp, a form-field label, a producer artifact) without being readable, and
# treating those few characters as "this document has text" would skip OCR and
# return almost nothing.
MIN_TEXT_LAYER_CHARS = 40

_notified_backends: set[str] = set()


def _notice(message: str) -> None:
    """One-time stderr-ish notice, in the same `[tag] message` shape
    agent/llm/router.py uses when it silently redirects a call."""
    if message not in _notified_backends:
        _notified_backends.add(message)
        print(f"[doc-intel] {message}")


class DocumentIntelligenceError(RuntimeError):
    """Raised for unrecoverable input problems (missing file, unreadable
    document, unusable page number). Callers that must not raise —
    `inspect_document` and the agent tool wrapper — catch this and return an
    "Error: ..." string, matching the convention in agent/tools.py."""


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════


class BoundingBox(BaseModel):
    """
    An axis-aligned rectangle in **normalized** page coordinates: 0.0 is the
    left/top edge, 1.0 the right/bottom edge. Normalized rather than pixel
    coordinates because the same box has to survive the page being rasterized
    at a different DPI than the one the extractor saw.

    Input is forgiving on purpose — VLM output is not a trusted, well-typed
    channel. All of these validate to the same box:

        BoundingBox(xmin=0.1, ymin=0.2, xmax=0.5, ymax=0.3)   # 0.0–1.0 floats
        BoundingBox(xmin=100, ymin=200, xmax=500, ymax=300)    # 0–1000 ints
        BoundingBox(xmin=0.5, ymin=0.3, xmax=0.1, ymax=0.2)    # reversed corners
        BoundingBox.from_vlm([200, 100, 300, 500])              # [ymin,xmin,ymax,xmax]

    Pixel coordinates are NOT auto-detected (they are indistinguishable from
    the 0–1000 convention) — use `from_pixels` when you have them.
    """

    model_config = ConfigDict(extra="ignore")

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @model_validator(mode="after")
    def _canonicalize(self) -> "BoundingBox":
        coords = [self.xmin, self.ymin, self.xmax, self.ymax]
        if max(abs(c) for c in coords) > _THOUSANDTHS_THRESHOLD:
            coords = [c / 1000.0 for c in coords]
        xmin, ymin, xmax, ymax = (min(1.0, max(0.0, c)) for c in coords)
        # Models routinely emit corners in the wrong order; a negative-width
        # rectangle would silently draw as nothing at all.
        self.xmin, self.xmax = min(xmin, xmax), max(xmin, xmax)
        self.ymin, self.ymax = min(ymin, ymax), max(ymin, ymax)
        return self

    # ─────────────────────────────────────────────── constructors

    @classmethod
    def from_vlm(cls, values: Sequence[float]) -> "BoundingBox":
        """
        Build from a vision model's list output in the standard
        **[ymin, xmin, ymax, xmax]** order — the convention Gemini and
        Qwen2.5-VL both emit, and the single most common source of silently
        transposed boxes when consumed as if it were [xmin, ymin, ...].
        """
        if len(values) != 4:
            raise DocumentIntelligenceError(
                f"expected 4 box coordinates in [ymin, xmin, ymax, xmax] order, got {values!r}")
        ymin, xmin, ymax, xmax = (float(v) for v in values)
        return cls(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    @classmethod
    def from_pixels(cls, box: Sequence[float], width: int, height: int) -> "BoundingBox":
        """Build from absolute pixel coordinates [xmin, ymin, xmax, ymax]."""
        x0, y0, x1, y1 = (float(v) for v in box)
        return cls(xmin=x0 / width, ymin=y0 / height, xmax=x1 / width, ymax=y1 / height)

    # ─────────────────────────────────────────────── geometry

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_pixels(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Absolute (x0, y0, x1, y1) for an image of the given size, guaranteed
        to be at least 1px on each axis so a thin box still renders."""
        x0 = int(round(self.xmin * width))
        y0 = int(round(self.ymin * height))
        x1 = max(x0 + 1, int(round(self.xmax * width)))
        y1 = max(y0 + 1, int(round(self.ymax * height)))
        return x0, y0, min(x1, width), min(y1, height)


class ExtractedField(BaseModel):
    """One grounded fact read off the document: what it is, what it says, how
    sure the extractor was, and where on the page it lives."""

    model_config = ConfigDict(extra="ignore")

    field_name: str
    field_value: Union[str, float]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    box: BoundingBox

    @field_validator("field_name")
    @classmethod
    def _clean_name(cls, v: str) -> str:
        cleaned = " ".join(str(v).split())
        if not cleaned:
            raise ValueError("field_name must not be empty")
        return cleaned

    @property
    def tier(self) -> str:
        """'high' | 'medium' | 'low' — drives the overlay colour and tells a
        reviewer how hard to look at this one."""
        if self.confidence >= CONFIDENCE_HIGH:
            return "high"
        if self.confidence >= CONFIDENCE_MEDIUM:
            return "medium"
        return "low"

    @property
    def color(self) -> Tuple[int, int, int]:
        return TIER_COLORS[self.tier]

    def overlay_label(self, max_chars: int = LABEL_MAX_CHARS) -> str:
        """The short "name  value" caption drawn above the box. Ellipsized for
        display only — the full value always stays intact in the JSON."""
        value = str(self.field_value)
        if len(value) > max_chars:
            value = value[: max_chars - 1].rstrip() + "…"
        return f"{self.field_name}  {value}"


class DocumentExtractionResult(BaseModel):
    """The full outcome of one page of one document: the validated fields, the
    extractor's untouched response (kept for debugging a bad extraction
    without re-running it), and the annotated image written to disk."""

    model_config = ConfigDict(extra="ignore")

    document_name: str
    page_number: int = Field(default=1, ge=1)
    fields: List[ExtractedField] = Field(default_factory=list)
    raw_json: dict = Field(default_factory=dict)
    annotated_image_path: str = ""
    # Not in the minimal spec, but every consumer immediately wants it: a
    # caller cannot judge a confidence score without knowing whether it came
    # from a vision model, an OCR engine, or the mock.
    backend: str = "mock"

    # ─────────────────────────────────────────────── conveniences

    def field_map(self) -> dict[str, Union[str, float]]:
        """Flat {name: value}. Later duplicates win, matching how a reader
        scanning top-to-bottom would resolve a repeated label."""
        return {f.field_name: f.field_value for f in self.fields}

    def low_confidence(self, threshold: float = CONFIDENCE_MEDIUM) -> List[ExtractedField]:
        """The fields a human actually has to check."""
        return [f for f in self.fields if f.confidence < threshold]

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def summary(self) -> str:
        """Human-readable digest for a terminal — one line per field, with the
        confidence tier spelled out so it survives a non-colour console."""
        if not self.fields:
            return f"{self.document_name} p{self.page_number}: no fields extracted."
        width = max(len(f.field_name) for f in self.fields)
        lines = [
            f"{self.document_name}  page {self.page_number}  "
            f"[{self.backend}]  {len(self.fields)} fields"
        ]
        for f in sorted(self.fields, key=lambda f: (f.box.ymin, f.box.xmin)):
            lines.append(
                f"  {f.field_name:<{width}}  {str(f.field_value):<24}"
                f"  {f.confidence:.2f} ({f.tier})"
            )
        if self.annotated_image_path:
            lines.append(f"  → annotated image: {self.annotated_image_path}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MOCK DOCUMENT — shared layout for the renderer AND the mock extractor
# ══════════════════════════════════════════════════════════════════════════════

# One row per extractable entity on the synthetic invoice.
#   name/value       what the mock extractor reports
#   conf             deliberately spread across all three tiers so a demo run
#                    exercises every overlay colour
#   box              normalized [xmin, ymin, xmax, ymax]
#   caption          the printed label on the rendered document (what a real
#                    invoice would say), drawn above the value by default
#   caption_x        set this to draw the caption INLINE at that x, as
#                    "CAPTION: value" on one line
#
# The renderer draws `value` INSIDE `box`, so the extractor's coordinates and
# the pixels genuinely correspond. Change a box here and the demo's overlay
# moves with it — there is no second copy to keep in sync.
#
# The mix of inline and stacked captions is deliberate, not decorative: real
# forms use both, and the `ocr` backend can only recover a field from the
# inline "Label: value" form. Having both on one page makes the difference
# between the backends visible in a single demo run instead of hiding it.
MOCK_INVOICE_LAYOUT: List[dict] = [
    # ── header ──
    {"name": "vendor_name",     "value": "ESDS Software Solution Ltd.", "conf": 0.98,
     "box": [0.055, 0.042, 0.520, 0.078], "caption": None, "size": "title"},
    {"name": "document_type",   "value": "TAX INVOICE",                 "conf": 0.99,
     "box": [0.660, 0.042, 0.945, 0.078], "caption": None, "size": "title", "align": "right"},
    # ── left column: parties ──
    {"name": "vendor_gstin",    "value": "27AABCE1234F1Z5",             "conf": 0.91,
     "box": [0.055, 0.112, 0.400, 0.139], "caption": "GSTIN"},
    {"name": "bill_to_name",    "value": "Nashik Municipal Corporation", "conf": 0.93,
     "box": [0.055, 0.196, 0.520, 0.223], "caption": "Bill To"},
    {"name": "bill_to_address", "value": "Rajiv Gandhi Bhavan, Nashik 422001", "conf": 0.66,
     "box": [0.055, 0.244, 0.560, 0.271], "caption": "Address"},
    {"name": "customer_gstin",  "value": "27AAALN0123C1ZK",             "conf": 0.54,
     "box": [0.055, 0.292, 0.400, 0.319], "caption": "Customer GSTIN"},
    # ── right column: document identity (inline "Label: value") ──
    {"name": "invoice_number",  "value": "INV-2024-0917",               "conf": 0.97,
     "box": [0.780, 0.112, 0.945, 0.139], "caption": "Invoice No.", "caption_x": 0.600},
    {"name": "invoice_date",    "value": "2024-09-17",                  "conf": 0.95,
     "box": [0.780, 0.160, 0.945, 0.187], "caption": "Invoice Date", "caption_x": 0.600},
    {"name": "due_date",        "value": "2024-10-17",                  "conf": 0.88,
     "box": [0.780, 0.208, 0.945, 0.235], "caption": "Due Date", "caption_x": 0.600},
    {"name": "po_number",       "value": "PO-77120",                    "conf": 0.71,
     "box": [0.780, 0.256, 0.945, 0.283], "caption": "PO Number", "caption_x": 0.600},
    # ── line items ──
    {"name": "line_1_description", "value": "Compute node CMP-M4 x12",  "conf": 0.94,
     "box": [0.055, 0.404, 0.520, 0.431], "caption": None},
    {"name": "line_1_amount",      "value": "1,00,800.00",              "conf": 0.90,
     "box": [0.730, 0.404, 0.945, 0.431], "caption": None, "align": "right"},
    {"name": "line_2_description", "value": "Block storage STO-SSD x40", "conf": 0.92,
     "box": [0.055, 0.452, 0.520, 0.479], "caption": None},
    {"name": "line_2_amount",      "value": "50,000.00",                "conf": 0.86,
     "box": [0.730, 0.452, 0.945, 0.479], "caption": None, "align": "right"},
    # ── totals (inline "Label: value") ──
    {"name": "subtotal",     "value": "1,50,800.00", "conf": 0.96,
     "box": [0.730, 0.566, 0.945, 0.593], "caption": "Subtotal", "caption_x": 0.545, "align": "right"},
    {"name": "tax_gst_18",   "value": "27,144.00",   "conf": 0.82,
     "box": [0.730, 0.620, 0.945, 0.647], "caption": "GST @ 18%", "caption_x": 0.545, "align": "right"},
    {"name": "total_due",    "value": "1,77,944.00", "conf": 0.99,
     "box": [0.730, 0.682, 0.945, 0.713], "caption": "TOTAL DUE (INR)",
     "caption_x": 0.545, "size": "bold", "align": "right"},
    {"name": "payment_terms", "value": "Net 30",     "conf": 0.58,
     "box": [0.235, 0.682, 0.385, 0.709], "caption": "Payment Terms", "caption_x": 0.055},
    # ── remittance / signature block ──
    {"name": "bank_account",  "value": "0234 5678 9012 3456", "conf": 0.79,
     "box": [0.185, 0.796, 0.520, 0.823], "caption": "Bank A/C", "caption_x": 0.055},
    {"name": "ifsc_code",     "value": "HDFC0001234",         "conf": 0.87,
     "box": [0.185, 0.850, 0.420, 0.877], "caption": "IFSC", "caption_x": 0.055},
    {"name": "authorised_signatory", "value": "S. Deshmukh",  "conf": 0.49,
     "box": [0.680, 0.850, 0.945, 0.877], "caption": "Authorised Signatory", "align": "right"},
]

MOCK_INVOICE_SIZE = (1000, 1400)   # portrait, ~A4 proportions at a readable raster size


def render_mock_invoice(size: Tuple[int, int] = MOCK_INVOICE_SIZE):
    """
    Draw a synthetic one-page invoice whose text sits exactly inside the boxes
    in MOCK_INVOICE_LAYOUT.

    This exists so the module is demonstrable with zero inputs: the demo
    script needs *a* document, and shipping a binary sample PDF into the repo
    to make a demo run is worse than generating one deterministically. It also
    makes the mock extractor honest — the boxes it returns are boxes around
    real rendered text, not arbitrary rectangles.
    """
    from PIL import Image, ImageDraw

    width, height = size
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    ink, muted, rule = (17, 24, 39), (107, 114, 128), (203, 213, 225)

    # Page chrome: accent bar, header rule, table header band, totals rule.
    draw.rectangle([0, 0, width, int(0.012 * height)], fill=(30, 64, 175))
    draw.line([(0.055 * width, 0.094 * height), (0.945 * width, 0.094 * height)],
              fill=rule, width=2)
    draw.rectangle([0.055 * width, 0.350 * height, 0.945 * width, 0.386 * height],
                   fill=(241, 245, 249))
    draw.line([(0.055 * width, 0.500 * height), (0.945 * width, 0.500 * height)],
              fill=rule, width=2)
    draw.line([(0.730 * width, 0.652 * height), (0.945 * width, 0.652 * height)],
              fill=ink, width=2)
    draw.line([(0.055 * width, 0.760 * height), (0.945 * width, 0.760 * height)],
              fill=rule, width=2)

    # Static table header — not an extracted field, just document furniture.
    header_font = _load_font(int(0.017 * height), bold=True)
    draw.text((0.065 * width, 0.358 * height), "DESCRIPTION", font=header_font, fill=muted)
    draw.text((0.740 * width, 0.358 * height), "AMOUNT (INR)", font=header_font, fill=muted)

    for item in MOCK_INVOICE_LAYOUT:
        x0, y0, x1, y1 = BoundingBox(
            xmin=item["box"][0], ymin=item["box"][1],
            xmax=item["box"][2], ymax=item["box"][3],
        ).to_pixels(width, height)

        if item.get("caption"):
            caption_font = _load_font(int(0.0145 * height), bold=False)
            if "caption_x" in item:
                # Inline: "CAPTION: value" on one baseline. The trailing colon
                # is what makes the field recoverable by the ocr backend.
                caption_x = item["caption_x"] * width
                caption = item["caption"].upper() + ":"
                available = x0 - caption_x - int(0.008 * width)
                caption_font = _fit_font(draw, caption, available,
                                          (y1 - y0) * 0.62, bold=False)
                # Baseline-align with the value rather than top-align: the
                # caption is a smaller font, and topping them out makes the
                # pair read as two separate lines to an OCR line grouper.
                _, _, _, caption_h = draw.textbbox((0, 0), caption, font=caption_font)
                draw.text((caption_x, y1 - caption_h - int(0.004 * height)),
                          caption, font=caption_font, fill=muted)
            else:
                draw.text((x0, y0 - int(0.020 * height)), item["caption"].upper(),
                          font=caption_font, fill=muted)

        style = item.get("size", "normal")
        # Fit the value to its box on BOTH axes, so the drawn glyphs stay
        # inside the rectangle the extractor will claim they occupy.
        bold = style in ("title", "bold")
        text = item["value"]
        font = _fit_font(draw, text, (x1 - x0),
                         (y1 - y0) * (0.82 if style == "title" else 0.74), bold=bold)
        # Right-align the money column; that is where it sits on a real
        # invoice, and a left-aligned amount inside a right-aligned box would
        # make the box look wrong even though it is correct.
        if item.get("align") == "right":
            text_width = draw.textlength(text, font=font)
            draw.text((x1 - text_width, y0), text, font=font, fill=ink)
        else:
            draw.text((x0, y0), text, font=font, fill=ink)

    footer_font = _load_font(int(0.0145 * height))
    draw.text((0.055 * width, 0.945 * height),
              "Synthetic document generated by Swarn doc_intelligence for demo purposes.",
              font=footer_font, fill=muted)
    return image


def mock_fields(schema_names: Optional[Iterable[str]] = None) -> List[ExtractedField]:
    """The mock extractor's output, optionally narrowed to a target schema's
    field names (so `target_schema` still changes the result offline)."""
    wanted = set(schema_names) if schema_names else None
    fields = []
    for item in MOCK_INVOICE_LAYOUT:
        if wanted is not None and item["name"] not in wanted:
            continue
        fields.append(ExtractedField(
            field_name=item["name"],
            field_value=item["value"],
            confidence=item["conf"],
            box=BoundingBox(xmin=item["box"][0], ymin=item["box"][1],
                            xmax=item["box"][2], ymax=item["box"][3]),
        ))
    return fields


# ══════════════════════════════════════════════════════════════════════════════
# FONTS
# ══════════════════════════════════════════════════════════════════════════════

# Ordered by preference. DejaVu ships with Pillow's own test assets and with
# essentially every Linux image; the rest cover macOS and Windows so an
# annotated image does not silently fall back to Pillow's 11px bitmap font
# (which cannot be scaled and looks broken on a 2000px page raster).
_FONT_CANDIDATES = {
    False: [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ],
    True: [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ],
}


@lru_cache(maxsize=64)
def _load_font(size: int, bold: bool = False):
    """A truetype font at `size`, or Pillow's default if none is installed.
    Cached because drawing an overlay asks for the same two or three sizes
    once per field."""
    from PIL import ImageFont

    size = max(8, int(size))
    for path in _FONT_CANDIDATES[bold]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, ValueError):
            continue
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1 scales this
    except TypeError:
        return ImageFont.load_default()


def _fit_font(draw, text: str, max_width: float, max_height: float, bold: bool = False):
    """
    Largest font that renders `text` within `max_width`.

    Sizing on height alone is not enough: "ESDS Software Solution Ltd." at a
    height-derived size is twice as wide as its box and runs straight through
    the field beside it, which would make the mock document's own text
    disagree with the coordinates the mock extractor reports for it.
    """
    size = max(8, int(max_height))
    while size > 8:
        font = _load_font(size, bold=bold)
        if draw.textlength(text, font=font) <= max_width:
            return font
        size -= 1
    return _load_font(8, bold=bold)


# ══════════════════════════════════════════════════════════════════════════════
# INSPECTOR
# ══════════════════════════════════════════════════════════════════════════════

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}

_VLM_SYSTEM_PROMPT = (
    "You are a document information-extraction engine. You read a document "
    "image and return ONLY structured JSON — no prose, no markdown fences, no "
    "explanation."
)

_VLM_USER_TEMPLATE = """\
Extract the key entities from this document image.

Return JSON of exactly this shape:
{{"fields": [{{"field_name": "<snake_case_name>", "field_value": "<verbatim text>", \
"confidence": <0.0-1.0>, "box_2d": [ymin, xmin, ymax, xmax]}}]}}

Rules:
- box_2d uses the standard normalized [ymin, xmin, ymax, xmax] order, each
  value an integer 0-1000 relative to the image's height/width.
- The box must tightly enclose the VALUE's text, not its printed label.
- field_value must be the text exactly as printed. Do not reformat, convert
  currencies, or normalize dates.
- confidence is your honest reading confidence. Use a low value when the text
  is blurred, cropped, handwritten, or ambiguous — a wrong confident answer is
  worse than an honest uncertain one.
- Omit any field you cannot actually see. Never invent a plausible value.
{schema_hint}"""


class DocumentInspector:
    """
    Stateless document → grounded-fields pipeline. Holds no document state
    between calls; the only instance state is configuration (where artifacts
    go, which backend to use, which VLM endpoint to talk to), so one inspector
    can be reused across documents or constructed per request, whichever fits
    the caller.

    Backend resolution, in order:
      1. the `backend` argument to `process_document`
      2. the `backend` passed to `__init__`
      3. $SWARN_DOC_BACKEND  ("vlm" | "ocr" | "mock" | "auto")
      4. "auto" → "vlm" if a VLM API key is configured, else "mock"
    """

    def __init__(
        self,
        artifacts_dir: Optional[Union[str, Path]] = None,
        backend: Optional[str] = None,
        vlm_model: Optional[str] = None,
        vlm_base_url: Optional[str] = None,
        vlm_api_key: Optional[str] = None,
        dpi: int = DEFAULT_PDF_DPI,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
        self.dpi = dpi

        # A dedicated set of VLM env vars rather than reusing router.py's
        # DEPLOYED_* trio: the deployed Swarn endpoint is a text model, and
        # sending it an image payload would fail at request time rather than
        # falling back cleanly. Vision is opt-in, explicitly.
        self.vlm_model    = vlm_model    or os.environ.get("SWARN_VLM_MODEL", "qwen2.5-vl-7b-instruct")
        self.vlm_base_url = vlm_base_url or os.environ.get("SWARN_VLM_BASE_URL", "")
        self.vlm_api_key  = vlm_api_key  or os.environ.get("SWARN_VLM_API_KEY", "")

        requested = backend or os.environ.get("SWARN_DOC_BACKEND", "auto")
        if requested not in BACKENDS + ("auto",):
            raise DocumentIntelligenceError(
                f"unknown backend {requested!r} — expected one of {', '.join(BACKENDS)} or 'auto'.")
        self.backend = requested

    # ─────────────────────────────────────────────── backend selection

    def _resolve_backend(self, file_path: str, override: Optional[str] = None) -> str:
        """
        Decide how to read this document.

        The one rule that matters: **`auto` never resolves to `mock`.** Mock is
        synthetic sample data; substituting it for a real document the user
        handed us would report another document's invoice number, vendor, and
        total as if they had been read off theirs. That is the exact
        "confident, plausible, wrong" failure this module exists to catch, and
        no amount of a `backend` field in the JSON makes it acceptable —
        nobody reads a provenance field to find out whether the values in
        front of them are real. Mock is therefore reachable ONLY by asking for
        it explicitly (`--backend mock` / `backend="mock"`), which the demo
        does for the synthetic invoice it generates itself.

        `auto` order, for a real document:
          1. vlm   — if VLM credentials are configured
          2. text  — if the PDF carries an extractable text layer. Preferred
                     over OCR because the characters are already exact and the
                     word boxes come from the PDF itself; OCR-ing a raster of
                     text we can read directly only adds transcription errors.
          3. ocr   — rasterize + tesseract, the only option for scans/images
          4. a clear, actionable error naming what to install or configure
        """
        requested = override or self.backend
        if requested != "auto":
            return requested

        if self.vlm_api_key:
            return "vlm"

        if _has_text_layer(file_path):
            _notice(f"no SWARN_VLM_API_KEY configured — reading {Path(file_path).name} "
                    "from its embedded PDF text layer (exact text, exact coordinates).")
            return "text"

        if _tesseract_available():
            _notice(f"no SWARN_VLM_API_KEY configured and {Path(file_path).name} has no "
                    "text layer — running local OCR (tesseract).")
            return "ocr"

        raise DocumentIntelligenceError(
            f"cannot read {Path(file_path).name}: it has no extractable text layer "
            "(so it is a scan or an image), and no extraction backend is available.\n"
            "Fix this by doing ONE of:\n"
            "  • install local OCR:  pip install pytesseract  AND  "
            "apt-get install tesseract-ocr   (macOS: brew install tesseract)\n"
            "  • configure a vision model:  export SWARN_VLM_API_KEY=...  "
            "[SWARN_VLM_BASE_URL=... SWARN_VLM_MODEL=...]\n"
            "Note: --backend mock returns SYNTHETIC sample data and must not be used "
            "as a stand-in for reading this document.")

    # ─────────────────────────────────────────────── input → images

    def convert_pdf_to_images(self, pdf_path: str, dpi: Optional[int] = None) -> List["Image.Image"]:
        """
        Rasterize every page of a PDF to a PIL image.

        pdf2image (poppler) is the preferred renderer and the one the module's
        dependency list names. It is optional, though, and pdfplumber — already
        a hard Phase 13 dependency — can rasterize through pypdfium2, so the
        fallback keeps this method working on a machine that never installed
        poppler. Both paths produce the same thing: RGB images, page order
        preserved.
        """
        dpi = dpi or self.dpi
        full = Path(pdf_path).expanduser().resolve()
        if not full.exists():
            raise DocumentIntelligenceError(f"file not found: {pdf_path}")

        try:
            from pdf2image import convert_from_path
        except ImportError:
            pass
        else:
            try:
                return [img.convert("RGB") for img in convert_from_path(str(full), dpi=dpi)]
            except Exception as exc:                                   # noqa: BLE001
                # Almost always "poppler not installed" — pdf2image imports
                # fine without the binaries it shells out to.
                _notice(f"pdf2image failed ({exc.__class__.__name__}: {exc}); "
                        "falling back to pdfplumber rasterization.")

        try:
            import pdfplumber
        except ImportError as exc:
            raise DocumentIntelligenceError(
                "PDF rendering needs 'pip install pdf2image' (plus the poppler "
                "binaries) or 'pip install pdfplumber'."
            ) from exc

        images: List["Image.Image"] = []
        try:
            with pdfplumber.open(str(full)) as pdf:
                for page in pdf.pages:
                    page_image = page.to_image(resolution=dpi)
                    # .original is the un-annotated raster; .annotated exists
                    # only after a draw_* call, so prefer the former.
                    pil = getattr(page_image, "original", None) or page_image.annotated
                    images.append(pil.convert("RGB"))
        except Exception as exc:                                       # noqa: BLE001
            raise DocumentIntelligenceError(f"could not rasterize {pdf_path}: {exc}") from exc
        return images

    def load_page_image(self, file_path: str, page_number: int = 1) -> Tuple["Image.Image", int]:
        """
        Return (image, total_pages) for one page of a PDF or for an image file.
        Images are treated as a single-page document, which keeps
        `process_document`'s flow identical for both input kinds.
        """
        from PIL import Image, UnidentifiedImageError

        full = Path(file_path).expanduser().resolve()
        if not full.exists():
            raise DocumentIntelligenceError(f"file not found: {file_path}")
        if page_number < 1:
            raise DocumentIntelligenceError(f"page_number must be >= 1, got {page_number}")

        if full.suffix.lower() == ".pdf":
            pages = self.convert_pdf_to_images(str(full))
            if not pages:
                raise DocumentIntelligenceError(f"{file_path} has no renderable pages.")
            if page_number > len(pages):
                raise DocumentIntelligenceError(
                    f"page {page_number} is out of range ({len(pages)} pages in {file_path}).")
            return pages[page_number - 1], len(pages)

        if full.suffix.lower() not in _IMAGE_SUFFIXES:
            raise DocumentIntelligenceError(
                f"unsupported document type '{full.suffix or '(none)'}' — expected a PDF "
                f"or one of: {', '.join(sorted(_IMAGE_SUFFIXES))}.")
        if page_number != 1:
            raise DocumentIntelligenceError(
                f"{full.name} is a single-page image; page {page_number} does not exist.")
        try:
            with Image.open(full) as handle:
                return handle.convert("RGB"), 1
        except UnidentifiedImageError as exc:
            raise DocumentIntelligenceError(f"{file_path} is not a readable image.") from exc

    # ─────────────────────────────────────────────── rendering

    def draw_bounding_boxes(
        self,
        image: "Image.Image",
        fields: List[ExtractedField],
        output_path: Union[str, Path],
    ) -> str:
        """
        Draw one rounded, confidence-coloured rectangle per field, each with a
        label tag naming the field and its value, and save the result.

        The boxes are drawn onto a transparent overlay that is alpha-composited
        at the end, rather than straight onto the page: the translucent fill
        has to let the underlying text show through (the whole point is to
        confirm the box sits over the right words), and compositing once keeps
        overlapping boxes from stacking their fills into an opaque smear.

        `output_path` may be relative, in which case it resolves inside the
        inspector's artifacts directory. Returns the absolute path written.
        """
        from PIL import Image, ImageDraw

        canvas = image.convert("RGBA")
        width, height = canvas.size
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        # Every dimension scales with the page raster so a 900px preview and a
        # 3000px 300-DPI scan get proportionally the same annotation weight.
        scale       = max(1.0, min(width, height) / 1000.0)
        stroke      = max(2, round(2.5 * scale))
        radius      = max(4, round(9 * scale))
        font        = _load_font(round(15 * scale), bold=True)
        pad_x       = max(4, round(6 * scale))
        pad_y       = max(2, round(4 * scale))
        gap         = max(2, round(4 * scale))

        ordered_fields = sorted(fields, key=lambda f: (f.box.ymin, f.box.xmin))
        # Every field's rectangle is an obstacle for every OTHER field's label
        # tag. Without this, a tag lands squarely on top of a neighbouring
        # field's value — which is worse than having no tag, because the box
        # it obscures is the evidence the reader came to check.
        all_boxes = [f.box.to_pixels(width, height) for f in ordered_fields]
        placed: List[Tuple[int, int, int, int]] = []

        # Top-to-bottom so the de-collision below resolves in reading order —
        # a tag pushed out of the way lands under the box above it, never over
        # a field that has not been drawn yet.
        for index, field in enumerate(ordered_fields):
            x0, y0, x1, y1 = all_boxes[index]
            color = field.color

            draw.rounded_rectangle(
                (x0, y0, x1, y1), radius=radius,
                outline=color + (255,), width=stroke, fill=color + (BOX_FILL_ALPHA,),
            )

            text = field.overlay_label()
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            tag_w = (right - left) + 2 * pad_x
            tag_h = (bottom - top) + 2 * pad_y

            obstacles = placed + [b for i, b in enumerate(all_boxes) if i != index]
            tag = self._place_tag(
                (x0, y0, x1, y1), tag_w, tag_h, gap, width, height, obstacles)
            placed.append(tag)

            draw.rounded_rectangle(tag, radius=max(2, radius // 2), fill=color + (TAG_FILL_ALPHA,))
            draw.text((tag[0] + pad_x - left, tag[1] + pad_y - top),
                      text, font=font, fill=(255, 255, 255, 255))

        annotated = Image.alpha_composite(canvas, overlay).convert("RGB")

        destination = Path(output_path)
        if not destination.is_absolute():
            destination = self.artifacts_dir / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(destination)
        return str(destination)

    @staticmethod
    def _place_tag(
        box: Tuple[int, int, int, int],
        tag_w: int,
        tag_h: int,
        gap: int,
        width: int,
        height: int,
        obstacles: Sequence[Tuple[int, int, int, int]],
    ) -> Tuple[int, int, int, int]:
        """
        Choose where a field's label tag goes: above the box by default, below
        it if there is no room above, and nudged downward as a last resort when
        both collide.

        `obstacles` is everything the tag must not cover — the tags already
        placed, AND every other field's box, since a tag printed over a
        neighbouring value hides the very evidence the overlay exists to show.

        Dense forms (an invoice header is four labels stacked in 15% of the
        page) otherwise produce tags printed on top of each other, which is
        strictly worse than an unlabelled box — the reviewer cannot tell which
        text belongs to which rectangle.
        """
        x0, y0, x1, y1 = box
        # Try the natural left-aligned position first, then flush-right of the
        # box: on a crowded page, shifting sideways often finds clear space
        # where every vertical position is blocked.
        x_options = [min(max(0, x0), max(0, width - tag_w)),
                     min(max(0, x1 - tag_w), max(0, width - tag_w))]
        y_options = [y0 - tag_h - gap, y1 + gap, y0 + gap]

        for tag_y in y_options:
            if tag_y < 0 or tag_y + tag_h > height:
                continue
            for tag_x in x_options:
                rect = (tag_x, tag_y, tag_x + tag_w, tag_y + tag_h)
                if not any(_overlaps(rect, other) for other in obstacles):
                    return rect

        # Everything collided: step down past the obstruction rather than
        # dropping the label entirely.
        tag_x = x_options[0]
        tag_y = min(max(0, y0), max(0, height - tag_h))
        rect = (tag_x, tag_y, tag_x + tag_w, tag_y + tag_h)
        while any(_overlaps(rect, other) for other in obstacles) and rect[3] + tag_h < height:
            rect = (rect[0], rect[1] + tag_h + gap, rect[2], rect[3] + tag_h + gap)
        return rect

    # ─────────────────────────────────────────────── main flow

    def process_document(
        self,
        file_path: str,
        target_schema: Optional[Type[BaseModel]] = None,
        page_number: int = 1,
        annotate: bool = True,
        output_path: Optional[Union[str, Path]] = None,
        backend: Optional[str] = None,
    ) -> DocumentExtractionResult:
        """
        Full pipeline for one page: load → extract → (validate) → annotate.

        target_schema, when given, is a pydantic model whose field names are
        the entities to look for. It does three things: it narrows the mock
        backend's output, it is injected into the VLM prompt as the requested
        shape, and the extracted {name: value} map is validated against it —
        with any validation failure recorded in `raw_json["schema_error"]`
        rather than raised. A schema mismatch means the *document* did not
        contain what the caller expected, which is a finding to report, not a
        crash: the boxes are still worth looking at.

        Raises DocumentIntelligenceError only for input problems (missing
        file, bad page number, unsupported type).
        """
        # Resolve the backend BEFORE rasterizing: if nothing can read this
        # document, the caller should hear that immediately rather than after
        # waiting for a 300-DPI render of a page nobody will extract from.
        chosen = self._resolve_backend(file_path, backend)
        image, total_pages = self.load_page_image(file_path, page_number)
        schema_names = _schema_field_names(target_schema)

        if chosen == "vlm":
            fields, raw = self._extract_with_vlm(image, schema_names)
        elif chosen == "text":
            fields, raw = self._extract_from_text_layer(file_path, page_number)
        elif chosen == "ocr":
            fields, raw = self._extract_with_ocr(image)
        elif chosen == "mock":
            fields, raw = self._extract_mock(schema_names)
        else:                                                          # pragma: no cover
            raise DocumentIntelligenceError(f"unknown backend {chosen!r}")

        # Belt and braces on the module's central promise. Every path above is
        # supposed to make this impossible; this asserts it at the one place
        # every path converges, so a future edit to backend selection cannot
        # quietly reintroduce synthetic values for a real document.
        if chosen in REAL_BACKENDS:
            _assert_no_mock_leak(fields, chosen, source=str(raw.get("source", "")))

        raw = dict(raw)
        raw["backend"] = chosen
        raw["page_count"] = total_pages
        raw["image_size"] = list(image.size)

        if target_schema is not None:
            raw["schema"] = target_schema.__name__
            try:
                validated = target_schema.model_validate(
                    {f.field_name: f.field_value for f in fields})
                raw["validated"] = validated.model_dump(mode="json")
            except Exception as exc:                                   # noqa: BLE001
                raw["schema_error"] = str(exc)

        document_name = Path(file_path).name
        annotated_path = ""
        if annotate:
            destination = output_path or _default_artifact_name(document_name, page_number)
            annotated_path = self.draw_bounding_boxes(image, fields, destination)

        return DocumentExtractionResult(
            document_name=document_name,
            page_number=page_number,
            fields=fields,
            raw_json=raw,
            annotated_image_path=annotated_path,
            backend=chosen,
        )

    def process_all_pages(
        self,
        file_path: str,
        target_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> List[DocumentExtractionResult]:
        """Run `process_document` over every page. Each page gets its own
        annotated image, so a 12-page contract produces 12 reviewable
        artifacts rather than one collage nobody can read."""
        _, total = self.load_page_image(file_path, 1)
        return [
            self.process_document(file_path, target_schema=target_schema,
                                  page_number=page, **kwargs)
            for page in range(1, total + 1)
        ]

    # ─────────────────────────────────────────────── backends

    def _extract_mock(
        self, schema_names: Optional[Iterable[str]] = None,
    ) -> Tuple[List[ExtractedField], dict]:
        """Deterministic sample invoice extraction — see MOCK_INVOICE_LAYOUT."""
        fields = mock_fields(schema_names)
        return fields, {
            "source": "mock",
            "note": ("Synthetic extraction. Configure SWARN_VLM_API_KEY for a real "
                     "vision model, or SWARN_DOC_BACKEND=ocr for local tesseract."),
            "fields": [
                {"field_name": f.field_name, "field_value": f.field_value,
                 "confidence": f.confidence,
                 "box_2d": [round(f.box.ymin * 1000), round(f.box.xmin * 1000),
                            round(f.box.ymax * 1000), round(f.box.xmax * 1000)]}
                for f in fields
            ],
        }

    def page_words(
        self, file_path: str, page_number: int = 1, backend: Optional[str] = None,
    ) -> Tuple[List[dict], Tuple[int, int], str]:
        """
        Read one page as POSITIONED WORDS: (words, (width, height), backend).

        This is the layer every real backend agrees on, and the seam other
        capabilities build against — doc_qa asks a question about a document by
        transcribing these words and mapping the answer's evidence back to
        their boxes. Extraction rules live above it; rendering lives below it;
        neither needs to know which backend produced the words.

        The `mock` backend has no words to give (it reads nothing), so asking
        for them is an error rather than a synthetic word list.
        """
        chosen = self._resolve_backend(file_path, backend)
        if chosen == "text":
            return (*self._text_layer_words(file_path, page_number), chosen)
        if chosen == "ocr":
            image, _ = self.load_page_image(file_path, page_number)
            return (*self._ocr_words(image), chosen)
        raise DocumentIntelligenceError(
            f"the '{chosen}' backend does not produce word-level text for a page. "
            "Use backend 'text' (PDF text layer) or 'ocr' (local tesseract).")

    def page_tables(self, file_path: str, page_number: int = 1) -> List[dict]:
        """
        Reconstruct one page's tables as CELLS WITH COORDINATES.

        Returns [{bbox, headers, rows: [[{column, row_index, col_index, text,
        bbox}, ...]]}] — the structure a table-aware evidence resolver needs and
        that neither pdfplumber's `extract()` (text only, no geometry) nor
        Phase 13's extract_pdf_structured (text rows, no geometry) provides.

        Why this reconstructs rather than trusting find_tables() directly
        ─────────────────────────────────────────────────────────────────
        A styled PDF table is full of geometry that is not table structure.
        On a real document this method was built against, find_tables() reports
        51 rows and 9 columns for a table a human reads as 12 rows and 3
        columns: the extra columns are the few-point padding strips between
        cell borders, and the extra rows are the hairline gaps between them.
        Trusting that shape directly produces mostly-empty cells and splits
        every wrapped cell in half.

        So the raw detection is used only for candidate BOUNDARIES, and the
        words decide what is real:

          columns  candidate x-ranges, minus any range that fully contains
                   another (a cell border nested inside an outer one), minus
                   any range no word actually falls in (pure padding)
          rows     candidate y-bands, taking the widest first and skipping any
                   that overlaps one already taken. The widest band spanning a
                   wrapped cell IS the logical row, so this merges
                   "Early Bird Idea Submission (before 24" / "Jul midnight)"
                   back into one cell instead of leaving two half-rows
          cells    words whose centre falls inside (row band × column range),
                   in reading order, with a bbox tight around those words

        Only PDFs have detectable ruling; an image returns [] and the caller
        falls back to line-based evidence, which is the pre-existing behaviour.
        """
        if Path(file_path).suffix.lower() != ".pdf":
            return []
        try:
            import pdfplumber
        except ImportError:
            return []

        try:
            with pdfplumber.open(str(Path(file_path).expanduser().resolve())) as pdf:
                if not 1 <= page_number <= len(pdf.pages):
                    return []
                page = pdf.pages[page_number - 1]
                words = page.extract_words() or []
                return [table for raw in page.find_tables()
                        if (table := _reconstruct_table(raw, words)) is not None]
        except Exception:                                              # noqa: BLE001
            # Table structure is an enrichment. A PDF whose ruling cannot be
            # parsed must still ingest for its text.
            return []

    def _text_layer_words(
        self, file_path: str, page_number: int,
    ) -> Tuple[List[dict], Tuple[int, int]]:
        """
        Read a PDF's own embedded text layer: exact characters, exact word
        boxes, straight from the file.

        Preferred over OCR whenever it is available. Rasterizing a digital PDF
        and transcribing the picture back into text can only lose information
        that is already sitting in the file — every OCR confusion (0/O, 1/l,
        rn/m) is a defect introduced by a step that was not necessary. The word
        boxes are in PDF points and are normalized against the page's own
        dimensions, so they overlay correctly on a raster at any DPI.
        """
        try:
            import pdfplumber
        except ImportError as exc:
            raise DocumentIntelligenceError(
                "reading a PDF text layer needs 'pip install pdfplumber'.") from exc

        with pdfplumber.open(str(Path(file_path).expanduser().resolve())) as pdf:
            if not 1 <= page_number <= len(pdf.pages):
                raise DocumentIntelligenceError(
                    f"page {page_number} is out of range ({len(pdf.pages)} pages in "
                    f"{Path(file_path).name}).")
            page = pdf.pages[page_number - 1]
            # fontname rides along with size: it is the only evidence a PDF
            # gives that a line is emphasised, and heading detection needs it
            # for documents that mark structure with weight rather than size.
            raw_words = page.extract_words(extra_attrs=["size", "fontname"]) or []
            page_size = (int(page.width), int(page.height))
            words = [{
                "text":   word["text"],
                "left":   float(word["x0"]),
                "top":    float(word["top"]),
                "width":  float(word["x1"]) - float(word["x0"]),
                "height": float(word["bottom"]) - float(word["top"]),
                # The text layer IS the document's text — there is no
                # transcription step to be uncertain about. Confidence in the
                # INTERPRETATION still comes from the rule that fired.
                "conf":   1.0,
                "size":   float(word.get("size") or 0) or None,
                "bold":   "bold" in str(word.get("fontname", "")).lower(),
            } for word in raw_words]
        return words, page_size

    def _extract_from_text_layer(
        self, file_path: str, page_number: int,
    ) -> Tuple[List[ExtractedField], dict]:
        """Entity extraction over the PDF's text layer — see _text_layer_words."""
        words, (page_width, page_height) = self._text_layer_words(file_path, page_number)
        fields, provenance = extract_entities(words, page_width, page_height)
        return fields, {
            "source": "pdf_text_layer",
            "n_words": len(words),
            "page_size": [page_width, page_height],
            "fields": provenance,
            "not_found": [name for name in COMMON_BUSINESS_FIELDS
                          if name not in {f.field_name for f in fields}],
        }

    def _extract_with_ocr(self, image: "Image.Image") -> Tuple[List[ExtractedField], dict]:
        """
        Local extraction via tesseract word boxes — the path for scans and
        images, which have no text layer to read.

        Every coordinate here is tesseract's own, and every value is a
        substring of what tesseract actually read; the per-word confidence it
        reports is carried through into each field's score, so a smudged line
        cannot come back looking as certain as a crisp one.
        """
        words, (width, height) = self._ocr_words(image)
        fields, provenance = extract_entities(words, width, height)
        return fields, {
            "source": "pytesseract",
            "n_words": len(words),
            "page_size": [width, height],
            "fields": provenance,
            "not_found": [name for name in COMMON_BUSINESS_FIELDS
                          if name not in {f.field_name for f in fields}],
        }

    @staticmethod
    def _ocr_words(image: "Image.Image") -> Tuple[List[dict], Tuple[int, int]]:
        """Tesseract word boxes for one page image, as positioned words."""
        try:
            import pytesseract
        except ImportError as exc:
            raise DocumentIntelligenceError(
                "the 'ocr' backend needs 'pip install pytesseract' plus the tesseract "
                "binary (apt-get install tesseract-ocr / brew install tesseract)."
            ) from exc
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        except Exception as exc:                                       # noqa: BLE001
            raise DocumentIntelligenceError(
                f"tesseract failed ({exc}). Is the tesseract binary installed and on "
                "PATH? (apt-get install tesseract-ocr / brew install tesseract)"
            ) from exc

        width, height = image.size
        words: List[dict] = []
        for i, raw_text in enumerate(data["text"]):
            text = raw_text.strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 0:                     # tesseract's marker for a non-word box
                continue
            words.append({
                "text": text,
                "left": float(data["left"][i]), "top": float(data["top"][i]),
                "width": float(data["width"][i]), "height": float(data["height"][i]),
                # tesseract reports 0-100; carry it through as the transcription
                # confidence rather than inventing a calibration we cannot justify.
                "conf": max(0.0, min(1.0, conf / 100.0)),
                "size": float(data["height"][i]),
            })
        return words, (width, height)

    def _extract_with_vlm(
        self,
        image: "Image.Image",
        schema_names: Optional[Iterable[str]] = None,
    ) -> Tuple[List[ExtractedField], dict]:
        """
        Extraction via an OpenAI-compatible vision endpoint.

        The `image_url` + base64 data-URL message shape below is what OpenAI,
        Gemini's compatibility layer, and every vLLM/SGLang server fronting a
        Qwen2.5-VL checkpoint all accept, so pointing SWARN_VLM_BASE_URL at any
        of them works without a per-provider adapter — the same reasoning that
        made agent/llm/router.py standardize on one OpenAI-compatible client.
        """
        if not self.vlm_api_key:
            raise DocumentIntelligenceError(
                "the 'vlm' backend needs SWARN_VLM_API_KEY (and SWARN_VLM_BASE_URL for "
                "a non-OpenAI endpoint). Leave them unset to use the mock backend.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise DocumentIntelligenceError(
                "the 'vlm' backend needs 'pip install openai'.") from exc

        hint = ""
        if schema_names:
            hint = ("\nExtract exactly these field_names, and no others: "
                    + ", ".join(schema_names))

        client_kwargs: dict[str, Any] = {"api_key": self.vlm_api_key}
        if self.vlm_base_url:
            client_kwargs["base_url"] = self.vlm_base_url
        client = OpenAI(**client_kwargs)

        try:
            response = client.chat.completions.create(
                model=self.vlm_model,
                messages=[
                    {"role": "system", "content": _VLM_SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text",
                         "text": _VLM_USER_TEMPLATE.format(schema_hint=hint)},
                        {"type": "image_url",
                         "image_url": {"url": _to_data_url(image)}},
                    ]},
                ],
                temperature=0,
                max_tokens=2048,
            )
            content = response.choices[0].message.content or ""
        except Exception as exc:                                       # noqa: BLE001
            raise DocumentIntelligenceError(
                f"VLM request to {self.vlm_base_url or 'the default OpenAI endpoint'} "
                f"failed: {exc}") from exc

        payload = parse_vlm_response(content)
        return fields_from_payload(payload), {"source": "vlm", "model": self.vlm_model,
                                              "response": payload}


# ══════════════════════════════════════════════════════════════════════════════
# PARSING HELPERS (module-level: unit-testable without an inspector or an image)
# ══════════════════════════════════════════════════════════════════════════════

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


# Reasoning models wrap their chain of thought in these before answering.
_THINK_BLOCK_RE = re.compile(
    r"<(think|thinking|reasoning)\b[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)


def parse_vlm_response(content: str) -> dict:
    """
    Pull the JSON object out of a vision model's reply.

    Models ignore "no markdown fences" instructions often enough that treating
    the reply as raw JSON is simply wrong in practice, so this strips a fenced
    block if present and otherwise falls back to the outermost {...} span. A
    reply with no recoverable JSON is an error, not an empty extraction —
    silently returning zero fields would look identical to "this document has
    no fields", which is a very different finding.
    """
    text = (content or "").strip()
    if not text:
        raise DocumentIntelligenceError("VLM returned an empty response.")

    # Reasoning models emit their chain of thought before the answer, often
    # inside <think> tags. The tags are not JSON and the reasoning inside them
    # routinely contains braces, so the outermost-{...} fallback below would
    # otherwise lock onto a fragment of the model's scratch work.
    text = _THINK_BLOCK_RE.sub("", text).strip()
    if not text:
        raise DocumentIntelligenceError(
            "the model returned only reasoning, with no answer after it.")

    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Outermost-span failed, which is what happens when reasoning prose sits
    # around the object and itself contains braces. Scan for balanced objects
    # and take the LAST one that parses: a model that sketches a draft answer
    # before the real one puts the real one last.
    for candidate in reversed(_balanced_objects(text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise DocumentIntelligenceError(
        f"VLM response was not valid JSON: {content[:200]!r}")


def _balanced_objects(text: str) -> List[str]:
    """Every brace-balanced {...} span in `text`, outermost only, in order.

    String-aware: a brace inside a JSON string literal is data, not nesting,
    and counting it would end every object at the wrong place.
    """
    spans, depth, start = [], 0, -1
    in_string = escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start != -1:
                spans.append(text[start:index + 1])
    return spans


def fields_from_payload(payload: Union[dict, list]) -> List[ExtractedField]:
    """
    Convert a parsed VLM payload into validated ExtractedFields.

    Tolerates the shapes models actually emit: a top-level list, {"fields":
    [...]}, or a single object; `box_2d` / `bbox` / `box` / `boundingBox` as
    either a 4-list in [ymin, xmin, ymax, xmax] order or a dict of named
    corners. Entries that are unusable (no box, wrong arity, non-numeric
    coordinates) are DROPPED rather than failing the whole page — one
    malformed field out of twenty should not cost the caller the other
    nineteen.
    """
    if isinstance(payload, dict):
        items = payload.get("fields") or payload.get("entities") or payload.get("data")
        if items is None:
            items = [payload] if "field_name" in payload else []
    else:
        items = payload
    if not isinstance(items, list):
        return []

    fields: List[ExtractedField] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_box = next((item[k] for k in ("box_2d", "bbox", "box", "boundingBox")
                        if k in item and item[k] is not None), None)
        if raw_box is None:
            continue
        try:
            if isinstance(raw_box, dict):
                box = BoundingBox.model_validate(raw_box)
            else:
                box = BoundingBox.from_vlm(list(raw_box))
            fields.append(ExtractedField(
                field_name=item.get("field_name") or item.get("name") or item.get("label"),
                field_value=item.get("field_value", item.get("value", "")),
                confidence=float(item.get("confidence", 1.0)),
                box=box,
            ))
        except Exception:                                              # noqa: BLE001
            continue
    return fields


# ══════════════════════════════════════════════════════════════════════════════
# ENTITY EXTRACTION — shared by the `text` and `ocr` backends
# ══════════════════════════════════════════════════════════════════════════════
#
# Both real local backends produce the same thing: a list of POSITIONED WORDS,
#   {"text", "left", "top", "width", "height", "conf" (0-1), "size"}
# in page pixels/points. Everything below works on that shape alone, so the
# entity rules are written once and a scan and a digital PDF get identical
# treatment — only the transcription confidence differs.
#
# THE INVARIANT, restated because it is the whole point of this module: every
# field emitted here is a substring of text that is physically present on the
# page, and its box is the union of the boxes of the words that substring came
# from. No rule may name a field the document does not support, and no rule may
# supply a value the document does not contain. Where a field is *expected* but
# absent, it is simply not emitted — `not_found` in raw_json records that it was
# looked for, which is a different statement from having found it.

# Rule confidence: how much to trust the INTERPRETATION, given that the text
# itself was read correctly. Multiplied by the per-word transcription
# confidence (1.0 for a PDF text layer, tesseract's own score for OCR), so a
# blurry scan and a crisp digital PDF do not claim the same certainty about
# the same inference.
RULE_CONFIDENCE = {
    "pattern":  0.95,   # a checksum-shaped match (GSTIN, IFSC, email) is near-proof
    "labelled": 0.90,   # "Label: value" on one line — the document states the pairing
    "stacked":  0.70,   # caption above value — a layout inference, not a stated pairing
    "heading":  0.75,   # largest text on the page
    "loose":    0.62,   # a bare date/amount with no label to anchor it
}

# Typed entity patterns. Each is (field_name, regex, kind) and matches against
# a whole reconstructed line, so the box can be narrowed to the matched span.
# Deliberately conservative: a pattern earns a slot here only if matching it is
# strong evidence on its own, because these fire without any label nearby.
_ENTITY_PATTERNS: List[Tuple[str, "re.Pattern", str]] = [
    # 27AABCE1234F1Z5 — state code, PAN, entity number, 'Z', checksum.
    ("gstin",       re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z\d]Z[A-Z\d]\b"), "pattern"),
    ("ifsc_code",   re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), "pattern"),
    ("pan",         re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "pattern"),
    ("email",       re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]*[a-zA-Z]{2,}\b"), "pattern"),
    ("phone",       re.compile(r"(?<!\d)(?:\+91[-\s]?)?[6-9]\d{9}(?!\d)"), "pattern"),
    ("website",     re.compile(r"\b(?:https?://|www\.)[\w.-]+\.[a-zA-Z]{2,}[\w/.-]*"), "pattern"),
    # Reference numbers: a case-insensitive PREFIX followed by a case-SENSITIVE
    # identifier that must contain a digit. Both halves of that are load-
    # bearing. An earlier version applied re.I to the whole pattern, and
    # "powered" duly matched as po_number ("po" + "wered") while "billed"
    # matched as invoice_number. The value was real text off the page, but the
    # NAME was fabricated — claiming a word is a PO number is exactly the kind
    # of invented field that makes an extraction untrustworthy. The lookahead
    # requiring a digit is what makes an identifier an identifier.
    ("invoice_number", re.compile(r"\b(?i:INV|INVOICE|BILL)(?i:\s*NO\.?)?[-/:\s]?(?=[A-Z0-9\-/]*\d)[A-Z0-9]+(?:[-/][A-Z0-9]+)*\b"), "pattern"),
    ("po_number",   re.compile(r"\b(?i:P\.?O\.?)(?i:\s*NO\.?)?[-/:\s]?(?=[A-Z0-9\-/]*\d)[A-Z0-9]+(?:[-/][A-Z0-9]+)*\b"), "pattern"),
    ("date",        re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{4})\b", re.I), "loose"),
    ("amount",      re.compile(r"(?:₹|Rs\.?|INR|\$)\s?\d[\d,]*(?:\.\d{1,2})?|\b\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?\b"), "loose"),
]

# Document-type keywords. Emitted as `document_type` only when one of these
# actually appears on the page — never assumed from context.
_DOCUMENT_TYPES = (
    "tax invoice", "proforma invoice", "commercial invoice", "invoice",
    "purchase order", "delivery challan", "credit note", "debit note",
    "receipt", "quotation", "statement of account", "bank statement",
    "salary slip", "payslip", "agreement", "contract", "certificate",
)

# Labels whose presence is worth reporting as "looked for, not found" when a
# document turns out not to have them. Reporting the absence is honest;
# inventing a plausible value is the failure mode this module exists to catch.
COMMON_BUSINESS_FIELDS = (
    "vendor_name", "document_type", "invoice_number", "invoice_date", "due_date",
    "gstin", "customer_name", "customer_address", "po_number", "subtotal",
    "tax", "total", "payment_terms", "bank_account", "ifsc_code",
    "authorised_signatory",
)

# A stacked caption must sit within this many multiples of its own height above
# its value, and start within this fraction of the page width of it, before the
# pair is believed. Loose enough for the generous line spacing of a slide or a
# form; the typography test below is what actually keeps false pairs out, so
# this does not have to be tight enough to do that job on its own.
STACKED_MAX_GAP_RATIO   = 4.0
STACKED_MAX_X_DRIFT     = 0.06
STACKED_MAX_LABEL_WORDS = 5
# A caption is set apart from its value typographically — smaller, or capitalised
# differently. Without this test, a LIST of Title Case items (a roster of names,
# a column of products) pairs each item with the next and invents a field per
# row: "Anupriya Raj" becomes the label of "Edunoori Spoorthi".
STACKED_SMALLER_FONT_RATIO = 0.92

_LABELISH_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 &/().'’#-]*$")
_HAS_LETTER_RE = re.compile(r"[A-Za-z]")


def _snake_case(label: str) -> str:
    """"Invoice No." → "invoice_no". Field names are keys; keeping the printed
    punctuation would make them awkward to address downstream."""
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", label.strip()).strip("_").lower()
    return re.sub(r"_+", "_", cleaned) or "field"


def _box_of(words: Sequence[dict], page_width: int, page_height: int) -> BoundingBox:
    """Union of the given words' boxes, normalized to the page."""
    left   = min(w["left"] for w in words)
    top    = min(w["top"] for w in words)
    right  = max(w["left"] + w["width"] for w in words)
    bottom = max(w["top"] + w["height"] for w in words)
    return BoundingBox.from_pixels((left, top, right, bottom), page_width, page_height)


def _mean_conf(words: Sequence[dict]) -> float:
    return sum(w.get("conf", 1.0) for w in words) / max(1, len(words))


def group_lines(words: Sequence[dict]) -> List[List[dict]]:
    """
    Group positioned words into visual lines by vertical overlap.

    Done geometrically rather than trusting a backend's own line numbering, so
    the PDF text layer and tesseract are grouped by identical rules — otherwise
    the two backends would disagree about what a "line" is and every downstream
    rule would need two variants.
    """
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (w["top"], w["left"]))
    lines: List[List[dict]] = [[ordered[0]]]
    for word in ordered[1:]:
        current = lines[-1]
        centre = word["top"] + word["height"] / 2
        ref = current[-1]
        ref_centre = ref["top"] + ref["height"] / 2
        # Same line if the vertical centres are within half a line height —
        # tolerant of the baseline jitter of mixed font sizes on one line.
        if abs(centre - ref_centre) <= max(ref["height"], word["height"]) * 0.5:
            current.append(word)
        else:
            lines.append([word])
    return [sorted(line, key=lambda w: w["left"]) for line in lines]


def _line_text_and_spans(words: Sequence[dict]) -> Tuple[str, List[Tuple[int, int]]]:
    """Reconstruct a line's text and each word's [start, end) offset in it, so a
    regex match can be traced back to the exact words — and therefore the exact
    box — it came from."""
    parts, spans, cursor = [], [], 0
    for word in words:
        text = word["text"]
        spans.append((cursor, cursor + len(text)))
        parts.append(text)
        cursor += len(text) + 1     # the joining space
    return " ".join(parts), spans


def _words_in_span(words: Sequence[dict], spans: Sequence[Tuple[int, int]],
                   start: int, end: int) -> List[dict]:
    return [w for w, (s, e) in zip(words, spans) if s < end and e > start]


def _is_labelish(text: str) -> bool:
    """Could this line be a field caption rather than content? A caption is a
    short noun phrase: no sentence punctuation, no verb-laden prose, few words."""
    stripped = text.strip().rstrip(":").strip()
    if not stripped or len(stripped.split()) > STACKED_MAX_LABEL_WORDS:
        return False
    if not _LABELISH_RE.match(stripped) or not _HAS_LETTER_RE.search(stripped):
        return False
    if len(stripped) > 40:
        return False
    # A caption is typically set apart typographically — all-caps or Title
    # Case. Requiring that keeps ordinary sentence fragments from pairing with
    # whatever happens to sit below them.
    words = stripped.split()
    return stripped.isupper() or all(w[:1].isupper() for w in words if w[:1].isalpha())


def extract_entities(
    words: Sequence[dict],
    page_width: int,
    page_height: int,
) -> Tuple[List[ExtractedField], List[dict]]:
    """
    Find business entities in a page of positioned words.

    Returns (fields, provenance) where provenance carries the per-field rule
    and transcription confidence — kept out of ExtractedField so the public
    schema stays exactly as specified, but recorded in raw_json so a
    questionable extraction can be explained without re-running it.

    Four rule families, applied to real words only:
      patterns  regex over each line, box narrowed to the matched span
      labelled  "Label: value" on one line (reusing multimodal_rag's _kv_match,
                which already rejects prose that merely contains a colon)
      stacked   a caption line directly above its value — how slides, forms,
                and most invoice headers are actually laid out
      heading   the largest text on the page, plus a document-type keyword
    """
    if not words:
        return [], []

    lines = group_lines(words)
    segments = [seg for line in lines for seg in _split_columns(line)]

    found: List[Tuple[ExtractedField, dict]] = []

    def emit(name: str, value: str, source_words: Sequence[dict], rule: str) -> None:
        value = " ".join(value.split())
        if not value or not source_words:
            return
        transcription = _mean_conf(source_words)
        confidence = max(0.0, min(1.0, transcription * RULE_CONFIDENCE[rule]))
        field = ExtractedField(
            field_name=name,
            field_value=value,
            confidence=confidence,
            box=_box_of(source_words, page_width, page_height),
        )
        found.append((field, {
            "field_name": name, "rule": rule,
            "rule_confidence": RULE_CONFIDENCE[rule],
            "transcription_confidence": round(transcription, 4),
        }))

    # ── 1. typed patterns, and 2. labelled pairs, both per segment ──
    for segment in segments:
        text, spans = _line_text_and_spans(segment)
        claimed: List[Tuple[int, int]] = []

        for name, pattern, rule in _ENTITY_PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.span()
                # A GSTIN contains a PAN; without this the same characters are
                # reported twice under two names.
                if any(s <= start and end <= e for s, e in claimed):
                    continue
                span_words = _words_in_span(segment, spans, start, end)
                if span_words:
                    claimed.append((start, end))
                    emit(name, match.group(0), span_words, rule)

        match = MultiModalIndexer_kv_match(text)
        if match:
            label, _value = match
            value_words = _words_after_colon(segment)
            if value_words:
                emit(_snake_case(label), " ".join(w["text"] for w in value_words),
                     value_words, "labelled")

        lowered = text.lower()
        for doc_type in _DOCUMENT_TYPES:
            if doc_type in lowered:
                start = lowered.index(doc_type)
                span_words = _words_in_span(segment, spans, start, start + len(doc_type))
                if span_words:
                    emit("document_type", text[start:start + len(doc_type)],
                         span_words, "heading")
                break

    # ── 3. stacked caption → value ──
    for caption, value_words in _stacked_pairs(segments, page_width):
        caption_text, _ = _line_text_and_spans(caption)
        value_text, _ = _line_text_and_spans(value_words)
        emit(_snake_case(caption_text), value_text, value_words, "stacked")

    # ── 4. dominant heading ──
    sizes = sorted(w.get("size") or w["height"] for w in words)
    median_size = sizes[len(sizes) // 2] or 1
    largest = max(sizes)
    if largest >= median_size * 1.6:
        title_words = [w for w in words if (w.get("size") or w["height"]) >= largest * 0.98]
        for line in group_lines(title_words):
            emit("document_title", " ".join(w["text"] for w in line), line, "heading")
            break

    return _dedupe_fields(found)


def _font_size(words: Sequence[dict]) -> float:
    return max((w.get("size") or w["height"]) for w in words)


def _looks_like_caption_of(caption: Sequence[dict], value: Sequence[dict]) -> bool:
    """
    Is `caption` typographically marked as the LABEL of `value`, rather than
    just the line that happens to sit above it?

    Two accepted signals, both of which a designer uses deliberately:
      * the caption is all-caps and the value is not (form and slide captions)
      * the caption is set in a visibly smaller font than the value

    Requiring one of them is what stops a column of similar-looking items — a
    team roster, a list of SKUs — from pairing row N as the label of row N+1.
    """
    caption_text = " ".join(w["text"] for w in caption).strip()
    value_text = " ".join(w["text"] for w in value).strip()
    if not caption_text or not value_text:
        return False
    if caption_text.isupper() and not value_text.isupper():
        return True
    return _font_size(caption) < _font_size(value) * STACKED_SMALLER_FONT_RATIO


def _stacked_pairs(
    segments: Sequence[Sequence[dict]], page_width: int,
) -> List[Tuple[Sequence[dict], Sequence[dict]]]:
    """
    Pair each caption segment with the nearest segment DIRECTLY BELOW IT IN THE
    SAME COLUMN.

    "The next segment in reading order" is not the same thing on a multi-column
    page, and using it silently drops the real pairs: on a two-column slide the
    segment after a left-column caption is usually the right column's next
    line, so the caption never meets its own value and the field is lost. This
    searches by geometry instead — same left edge, smallest positive vertical
    gap — which is what "below it" actually means.
    """
    pairs: List[Tuple[Sequence[dict], Sequence[dict]]] = []
    for caption in segments:
        caption_text, _ = _line_text_and_spans(caption)
        if not _is_labelish(caption_text):
            continue
        caption_left = min(w["left"] for w in caption)
        caption_bottom = max(w["top"] + w["height"] for w in caption)
        line_height = max(w["height"] for w in caption)

        best, best_gap = None, None
        for candidate in segments:
            if candidate is caption:
                continue
            gap = min(w["top"] for w in candidate) - caption_bottom
            # Small negatives allow for a caption and value whose boxes just
            # touch (tight leading), without letting the line ABOVE qualify.
            if not (-line_height * 0.2 <= gap <= line_height * STACKED_MAX_GAP_RATIO):
                continue
            if abs(min(w["left"] for w in candidate) - caption_left) > STACKED_MAX_X_DRIFT * page_width:
                continue
            if best_gap is None or gap < best_gap:
                best, best_gap = candidate, gap

        if best is None or not _looks_like_caption_of(caption, best):
            continue
        pairs.append((caption, best))
    return pairs


def _dedupe_fields(
    found: Sequence[Tuple[ExtractedField, dict]],
) -> Tuple[List[ExtractedField], List[dict]]:
    """
    Collapse duplicates, keeping the most confident reading of each.

    Two rules fire on the same text often and legitimately — "Invoice No.:
    INV-1" matches both the labelled rule and the invoice_number pattern. That
    is one fact, and reporting it twice would double-count it in the summary
    and draw two boxes on top of each other.
    """
    best: dict[tuple, Tuple[ExtractedField, dict]] = {}
    for field, provenance in found:
        key = (field.field_name, str(field.field_value).lower())
        existing = best.get(key)
        if existing is None or field.confidence > existing[0].confidence:
            best[key] = (field, provenance)

    # Same name, and one value contains the other: two rules read the same fact
    # at different widths. "TEAM CAPTAIN" stacked over "Team Captain: Sargam
    # Maurya" yields the whole line, while the labelled rule yields just
    # "Sargam Maurya" — the same field, and the tighter, more confident reading
    # is the one worth keeping (it also draws the more useful box).
    by_name: dict[str, List[Tuple[ExtractedField, dict]]] = {}
    for pair in best.values():
        by_name.setdefault(pair[0].field_name, []).append(pair)

    survivors: List[Tuple[ExtractedField, dict]] = []
    for candidates in by_name.values():
        candidates.sort(key=lambda pair: pair[0].confidence, reverse=True)
        for pair in candidates:
            value = str(pair[0].field_value).lower()
            if any(value in str(kept[0].field_value).lower()
                   or str(kept[0].field_value).lower() in value
                   for kept in survivors if kept[0].field_name == pair[0].field_name):
                continue
            survivors.append(pair)

    ordered = sorted(survivors, key=lambda pair: (pair[0].box.ymin, pair[0].box.xmin))

    # A second pass on VALUE alone: the same string found by two different
    # rules under two different names (a stacked "team_name" and a labelled
    # "name") is still one piece of text on the page, and one box.
    seen_values: dict[str, float] = {}
    kept: List[Tuple[ExtractedField, dict]] = []
    for field, provenance in ordered:
        value_key = str(field.field_value).lower()
        if seen_values.get(value_key, -1.0) >= field.confidence:
            continue
        seen_values[value_key] = field.confidence
        kept = [pair for pair in kept if str(pair[0].field_value).lower() != value_key]
        kept.append((field, provenance))

    kept.sort(key=lambda pair: (pair[0].box.ymin, pair[0].box.xmin))

    # An unlabelled date/amount sitting INSIDE a richer field's box is the same
    # text read twice at two widths — "9 August 2026" inside submit_by's
    # "9 August 2026, 10:00 PM". Keep the one that carries the label.
    def _contained(inner: ExtractedField, outer: ExtractedField) -> bool:
        return (outer.box.xmin <= inner.box.xmin and inner.box.xmax <= outer.box.xmax
                and outer.box.ymin <= inner.box.ymin and inner.box.ymax <= outer.box.ymax)

    loose_names = {name for name, _, rule in _ENTITY_PATTERNS if rule == "loose"}
    kept = [
        pair for pair in kept
        if not (pair[0].field_name in loose_names and any(
            other[0] is not pair[0]
            and str(pair[0].field_value).lower() in str(other[0].field_value).lower()
            and _contained(pair[0], other[0])
            for other in kept))
    ]

    # A page can legitimately hold several amounts or dates. Numbering them
    # keeps every one addressable — an unsuffixed duplicate name silently
    # loses all but the last when the fields are read into a {name: value} map.
    counts: dict[str, int] = {}
    for field, _ in kept:
        counts[field.field_name] = counts.get(field.field_name, 0) + 1
    seen: dict[str, int] = {}
    for field, provenance in kept:
        if counts[field.field_name] > 1:
            seen[field.field_name] = seen.get(field.field_name, 0) + 1
            numbered = f"{field.field_name}_{seen[field.field_name]}"
            provenance["field_name"] = numbered
            field.field_name = numbered

    return [pair[0] for pair in kept], [pair[1] for pair in kept]


def MultiModalIndexer_kv_match(text: str):
    """Thin indirection over multimodal_rag's key/value line parser.

    Imported lazily and wrapped here so the entity engine stays importable —
    and unit-testable — without pulling `agent.*` in at module load, per the
    one-way wiring rule in swarn/__init__.py."""
    from agent.multimodal_rag import MultiModalIndexer
    return MultiModalIndexer._kv_match(text)


# ── table reconstruction (see DocumentInspector.page_tables) ──────────────
# A column range narrower than this is a cell border's padding strip, not a
# column. Real columns hold words; these never do.
MIN_TABLE_COLUMN_WIDTH = 8.0
# A "table" of one cell is a styled callout box, not tabular data — treating
# it as a table would invite row/cell matches against ordinary prose.
MIN_TABLE_COLUMNS = 2
MIN_TABLE_ROWS = 2


def _minimal_ranges(ranges: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Drop any range that fully contains another.

    Bordered cells nest: an outer (45.6, 241.0) around an inner (51.8, 234.4).
    Both are reported as columns, and keeping both duplicates every cell. The
    inner one is the actual text column, so containers lose."""
    return [r for r in ranges
            if not any(other != r and r[0] <= other[0] and other[1] <= r[1]
                       for other in ranges)]


def _non_overlapping_bands(
    bands: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Keep widest-first, skipping any band overlapping one already kept.

    A wrapped cell produces a tall band covering all its lines PLUS a nested
    band for the continuation line alone. The tall one is the logical row, so
    preferring it keeps multi-line cells intact."""
    kept: List[Tuple[float, float]] = []
    for band in sorted(bands, key=lambda b: (b[0], -(b[1] - b[0]))):
        if not any(band[0] < other[1] and other[0] < band[1] for other in kept):
            kept.append(band)
    return sorted(kept)


def _words_in_rect(
    words: Sequence[dict], x_range: Tuple[float, float], y_range: Tuple[float, float],
) -> List[dict]:
    """Words whose CENTRE lies in the rectangle, in reading order.

    Centre rather than overlap: a word straddling a cell border belongs to the
    cell holding most of it, and overlap-based assignment would put it in both.
    """
    inside = [
        word for word in words
        if y_range[0] - 1 <= word["top"] <= y_range[1] + 1
        and x_range[0] - 0.5 <= (word["x0"] + word["x1"]) / 2 <= x_range[1] + 0.5
    ]
    return sorted(inside, key=lambda word: (word["top"], word["x0"]))


def _reconstruct_table(raw, words: Sequence[dict]) -> Optional[dict]:
    """One detected table → cells with text and tight boxes, or None if what
    was detected is not really tabular."""
    x0, y0, x1, y1 = raw.bbox
    inside = [w for w in words if y0 - 1 <= w["top"] <= y1 + 1]
    if not inside:
        return None

    candidates = {(round(cell[0], 1), round(cell[2], 1))
                  for row in raw.rows for cell in row.cells
                  if cell and cell[2] - cell[0] > MIN_TABLE_COLUMN_WIDTH}
    columns = [column for column in _minimal_ranges(sorted(candidates))
               if _words_in_rect(inside, column, (y0, y1))]

    band_candidates = [(round(row.bbox[1], 1), round(row.bbox[3], 1)) for row in raw.rows]
    bands = [band for band in _non_overlapping_bands(band_candidates)
             if _words_in_rect(inside, (x0, x1), band)]

    if len(columns) < MIN_TABLE_COLUMNS or len(bands) < MIN_TABLE_ROWS:
        return None

    rows: List[List[dict]] = []
    for row_index, band in enumerate(bands):
        cells = []
        for col_index, column in enumerate(columns):
            cell_words = _words_in_rect(inside, column, band)
            text = " ".join(word["text"] for word in cell_words)
            if cell_words:
                bbox = [min(w["x0"] for w in cell_words), min(w["top"] for w in cell_words),
                        max(w["x1"] for w in cell_words), max(w["bottom"] for w in cell_words)]
            else:
                # An empty cell still needs a box so a row's geometry is
                # complete; the cell rectangle is the honest answer.
                bbox = [column[0], band[0], column[1], band[1]]
            cells.append({
                "row_index": row_index, "col_index": col_index,
                "text": text, "bbox": [round(v, 2) for v in bbox],
            })
        rows.append(cells)

    # The first row is a header row only if every cell in it has text — a
    # partially-filled first row is data, and naming columns after it would
    # attach wrong labels to every cell below.
    headers = [cell["text"] for cell in rows[0]]
    has_headers = all(header.strip() for header in headers)
    if has_headers:
        for row in rows:
            for cell, header in zip(row, headers):
                cell["column"] = header
    else:
        headers = []
        for row in rows:
            for cell in row:
                cell["column"] = f"col{cell['col_index'] + 1}"

    return {
        "bbox": [round(v, 2) for v in (x0, y0, x1, y1)],
        "headers": headers,
        "rows": rows,
    }


COLUMN_GAP_RATIO = 1.6   # a horizontal gap this many text-heights wide starts a new column

# ── page-level column bands (see column_bands) ────────────────────────────
# Two segments belong to different COLUMNS OF THE PAGE when their left edges
# are at least this far apart, as a fraction of page width. Measured: a
# two-column slide puts its columns at x≈3% and x≈64%, a 60-point separation.
#
# This threshold is NOT what keeps tables safe, and it cannot be — a wide
# table's own columns can sit 33% apart, further than some page gutters, so no
# single distance separates the two cases. Tables are protected structurally
# instead: the caller only consults these bands on a page with no detected
# table (see doc_store._lines_for_page). Keep that guard; do not try to
# replace it by tuning this number.
COLUMN_BAND_GAP = 0.25

# Two segments on ONE line that are further apart than this (fraction of page
# width) are page columns rather than cells of a row. Consulted ONLY when no
# table was detected around them — see doc_store._same_row for why both signals
# are needed.
#
# 0.40 sits in a measured gap, not at a guess. Across two real documents:
#   borderless table cells   2%, 3%, 5%, 29%, 33%   <- must stay joined
#   page gutters             46%, 49%, 49%, 53%, 54% <- must split
# Nothing was observed between 33% and 46%. Widely-spaced TABLE columns (43%
# in a contacts table) land above this line, and are held together by the
# detected-table signal instead — which is the reason that signal is primary
# and this one is only the fallback.
COLUMN_JOIN_GAP = 0.40

# Minimum words on BOTH sides of a split before the two sides are treated as
# running prose rather than table cells.
#
# The distance rule above assumes page gutters are wide. On a UPPCL electricity
# bill they are not: its two-column notice block leaves a 4-6% gutter, which
# lands *inside* the borderless-cell range (2-5%) that COLUMN_JOIN_GAP exists
# to rescue. Distance cannot separate those two cases, and neither can
# alignment or baseline drift — that bill's columns share left edges and
# baselines exactly, like a table's.
#
# Content length can. A table cell is a value ("24 Jul", "Idea submission
# closes"); a page column is a sentence. 6 sits above the longest cell in the
# borderless fixtures (3 words) and well below the bill's columns (11-14).
# Used only as the second of two signals — see doc_store._same_row.
PROSE_SEGMENT_WORDS = 6


def column_bands(lefts: Sequence[float], page_width: float) -> List[Tuple[float, float]]:
    """
    Cluster segment left edges into page-level column bands.

    Returns the (min_left, max_left) of each band, left to right. One band —
    the common case for an ordinary business document — means the page is not
    multi-column and nothing downstream should change.

    Clustering on left EDGES rather than on empty vertical gutters is
    deliberate. A gutter-based scan is defeated by any full-width element: on
    the slide that motivated this, a title spanning 26..495pt bridges the
    gutter and collapses the two columns into one region. Left edges are
    unaffected, because a designer aligns a column's contents to a shared left
    edge — which is exactly what makes them a column.
    """
    ordered = sorted(lefts)
    if not ordered:
        return []
    bands: List[List[float]] = [[ordered[0]]]
    for previous, current in zip(ordered, ordered[1:]):
        if (current - previous) / max(page_width, 1.0) >= COLUMN_BAND_GAP:
            bands.append([current])
        else:
            bands[-1].append(current)
    return [(min(band), max(band)) for band in bands]


def band_of(left: float, bands: Sequence[Tuple[float, float]]) -> int:
    """Index of the band a segment starting at `left` belongs to.

    A full-width element is placed by its left edge, which puts it at the top
    of the leftmost column it starts in — where a reader would encounter it.
    """
    if not bands:
        return 0
    # Nearest band, not strict containment. Real layouts are not pixel-perfect:
    # on the slide that motivated this, the caption "SUBMIT BY" starts at
    # x=613.6 while the rest of its column aligns at x=620. Strict containment
    # exiles it to the other column, stranding it far from the value it labels.
    def distance(band):
        low, high = band
        return 0.0 if low <= left <= high else min(abs(left - low), abs(left - high))
    return min(range(len(bands)), key=lambda i: distance(bands[i]))


def _split_columns(words: List[dict]) -> List[List[dict]]:
    """
    Split one OCR text line into left-to-right column segments.

    Tesseract's block/paragraph/line numbering does not reliably separate side-
    by-side columns: on a two-column invoice header it happily returns
    "Nashik Municipal Corporation DUE DATE: 2024-10-17" as a single line, which
    then parses as a field named after the *other* column's text. A wide
    horizontal gap is the signal a human reads as a column break, so use the
    same cue — a gap wider than COLUMN_GAP_RATIO × the line's own text height,
    which scales automatically with font size instead of assuming a page width.

    The one exception is a gap that follows a colon. Forms routinely set the
    label at a tab stop and the value flush right ("INVOICE NO.:" … "INV-2024-
    0917"), so a wide gap there separates a label from its OWN value, not two
    columns — splitting on it would throw away every right-aligned field on
    the page.
    """
    if not words:
        return []
    ordered = sorted(words, key=lambda w: w["left"])
    heights = sorted(w["height"] for w in ordered)
    text_height = heights[len(heights) // 2] or 1
    threshold = COLUMN_GAP_RATIO * text_height

    segments: List[List[dict]] = [[ordered[0]]]
    for previous, word in zip(ordered, ordered[1:]):
        gap = word["left"] - (previous["left"] + previous["width"])
        if gap > threshold and not previous["text"].rstrip().endswith(":"):
            segments.append([word])
        else:
            segments[-1].append(word)
    return segments


def _words_after_colon(words: List[dict]) -> List[dict]:
    """The OCR words that make up the VALUE half of a "Label: value" line.

    Everything up to and including the word carrying the colon is the label.
    A word like "Date:2024-09-17" (no space after the colon) still counts as a
    value word, since dropping it would leave the field with no value at all."""
    for i, word in enumerate(words):
        if ":" in word["text"]:
            tail = word["text"].split(":", 1)[1].strip()
            return ([word] if tail else []) + words[i + 1:]
    return []


def _has_text_layer(file_path: str, min_chars: int = MIN_TEXT_LAYER_CHARS) -> bool:
    """
    Does this PDF carry text we can read directly, or is it a picture of text?

    Only the first few pages are probed: a scan is a scan throughout, and
    opening every page of a 300-page document to answer a routing question
    would cost more than the extraction it is routing.
    """
    path = Path(file_path).expanduser()
    if path.suffix.lower() != ".pdf":
        return False                       # an image never has a text layer
    try:
        import pdfplumber
    except ImportError:
        return False
    try:
        with pdfplumber.open(str(path.resolve())) as pdf:
            for page in pdf.pages[:3]:
                if len((page.extract_text() or "").strip()) >= min_chars:
                    return True
    except Exception:                                                  # noqa: BLE001
        # A malformed/encrypted PDF is not a routing decision to make here —
        # let the chosen backend report the real failure.
        return False
    return False


def _tesseract_available() -> bool:
    """True only if BOTH pytesseract and the tesseract binary are present.
    The Python package installs happily without the binary, so importing it is
    not evidence that OCR can actually run."""
    try:
        import pytesseract
    except ImportError:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:                                                  # noqa: BLE001
        return False


# The mock layout's hard-coded rectangles. A real extractor derives every box
# from the document — from PDF word geometry, OCR word geometry, or a model's
# output — so reproducing these exact constants to full float precision is not
# something that happens by chance. That makes box identity a sound leak
# signal, where value identity is not: OCR run over the mock invoice IMAGE
# legitimately reads "1,77,944.00" off it, because that string really is
# printed there. Comparing values would flag a correct read as a leak.
_MOCK_BOXES = frozenset(
    (round(item["box"][0], 6), round(item["box"][1], 6),
     round(item["box"][2], 6), round(item["box"][3], 6))
    for item in MOCK_INVOICE_LAYOUT
)


def _assert_no_mock_leak(
    fields: Sequence[ExtractedField], backend: str, source: str = "",
) -> None:
    """
    Guard the module's central promise: a real backend reports only what it
    read off the supplied document.

    Two independent checks, because the regression this exists to prevent was
    a dispatch-wiring mistake and either symptom alone is enough to catch it:
      * the extractor's own provenance tag says "mock"
      * several fields carry the mock layout's literal coordinates

    A single coordinate collision is conceivable on a contrived page, so the
    box check needs a few before it fires; the whole synthetic layout coming
    back at once is what a real leak looks like.
    """
    if source == "mock":
        raise DocumentIntelligenceError(
            f"internal error: the '{backend}' backend returned output tagged as coming "
            "from the synthetic mock extractor. Refusing to report synthetic data as "
            "this document's contents.")

    boxes = {(round(f.box.xmin, 6), round(f.box.ymin, 6),
              round(f.box.xmax, 6), round(f.box.ymax, 6)) for f in fields}
    overlap = boxes & _MOCK_BOXES
    if len(overlap) > 2:
        raise DocumentIntelligenceError(
            f"internal error: the '{backend}' backend returned {len(overlap)} fields at the "
            "synthetic mock invoice's hard-coded coordinates. Refusing to report synthetic "
            "data as this document's contents.")


def _overlaps(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _to_data_url(image: "Image.Image", fmt: str = "PNG") -> str:
    """PIL image → `data:image/png;base64,...`, the inline-image form every
    OpenAI-compatible vision endpoint accepts."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def _schema_field_names(schema: Optional[Type[BaseModel]]) -> Optional[List[str]]:
    if schema is None:
        return None
    return list(schema.model_fields.keys())


def _default_artifact_name(document_name: str, page_number: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(document_name).stem) or "document"
    return f"{stem}_p{page_number}_annotated.png"


# ══════════════════════════════════════════════════════════════════════════════
# AGENT / CLI ENTRY POINTS
# ══════════════════════════════════════════════════════════════════════════════


def inspect_document(
    file_path: str,
    page_number: int = 1,
    annotate: bool = True,
    backend: Optional[str] = None,
    artifacts_dir: Optional[Union[str, Path]] = None,
    include_raw: bool = False,
) -> dict:
    """
    The registration surface: everything the agent tool (`swarn_doc_inspect`)
    and the CLI (`swarn doc-inspect`) need, as a plain JSON-serializable dict.

    Never raises for an input problem — a bad path or page returns
    `{"error": "..."}`, which agent/tools.py turns into the "Error: ..." string
    the agent loop knows how to recover from.

    `include_raw` is off by default because raw_json holds the extractor's
    complete response, and flooding an agent's context with a second copy of
    the same fields is a real cost for a debugging aid it rarely needs.

    Backend selection is the inspector's (`auto` never yields mock), so an
    agent calling this on a user's document cannot be handed synthetic values.
    Asking for mock explicitly still works and is reported in `backend`, with
    `synthetic: true` alongside it so a caller cannot miss it.
    """
    try:
        inspector = DocumentInspector(artifacts_dir=artifacts_dir, backend=backend)
        result = inspector.process_document(
            file_path, page_number=page_number, annotate=annotate)
    except DocumentIntelligenceError as exc:
        return {"error": str(exc)}

    payload = result.to_dict()
    if result.backend not in REAL_BACKENDS:
        payload["synthetic"] = True
        payload["warning"] = (
            "backend 'mock' returns SYNTHETIC SAMPLE DATA — these values were NOT "
            "read from the supplied document and must not be reported as its contents.")
    if not include_raw:
        # Keep the cheap provenance keys; drop the bulky echo of the response.
        payload["raw_json"] = {
            k: v for k, v in result.raw_json.items()
            if k in ("backend", "page_count", "image_size", "schema", "schema_error", "note")
        }
    payload["n_fields"] = len(result.fields)
    payload["n_low_confidence"] = len(result.low_confidence())
    return payload


def create_mock_document(
    destination: Optional[Union[str, Path]] = None,
    size: Tuple[int, int] = MOCK_INVOICE_SIZE,
) -> str:
    """Render the synthetic invoice to disk and return its path — used by the
    demo script, and handy for anyone who wants a document to point the OCR or
    VLM backends at without hunting for a real invoice."""
    destination = Path(destination) if destination else ARTIFACTS_DIR / "mock_invoice.png"
    if not destination.is_absolute():
        destination = ARTIFACTS_DIR / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    render_mock_invoice(size).save(destination)
    return str(destination)


__all__ = [
    "BoundingBox",
    "ExtractedField",
    "DocumentExtractionResult",
    "DocumentInspector",
    "DocumentIntelligenceError",
    "inspect_document",
    "create_mock_document",
    "render_mock_invoice",
    "mock_fields",
    "parse_vlm_response",
    "fields_from_payload",
    "ARTIFACTS_DIR",
    "MOCK_INVOICE_LAYOUT",
    "CONFIDENCE_HIGH",
    "CONFIDENCE_MEDIUM",
]
