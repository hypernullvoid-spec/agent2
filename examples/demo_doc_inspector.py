#!/usr/bin/env python3
"""
Demo: Visual Document Intelligence & Bounding-Box Inspector.

Run it with no arguments and it works — no API key, no GPU, no network, no
sample file to go find:

    python examples/demo_doc_inspector.py

That renders a synthetic invoice, extracts its fields with the deterministic
mock backend, draws confidence-coloured bounding boxes over the values, writes
artifacts/annotated_doc_sample.png, and prints the structured JSON.

Point it at a real document and it reads THAT document — never the mock:

    python examples/demo_doc_inspector.py invoice.pdf
    python examples/demo_doc_inspector.py scan.png --backend ocr

Backends (see swarn/capabilities/doc_intelligence.py for the full contract):
    text   the PDF's own embedded text layer — exact text, exact word boxes
    ocr    local tesseract word boxes, for scans and images
    vlm    OpenAI-compatible vision endpoint  (SWARN_VLM_API_KEY=...)
    mock   SYNTHETIC sample data. Used automatically ONLY for the invoice this
           script generates itself. It is never selected for a document you
           supply — reporting one document's values as another's is the exact
           failure this capability exists to catch.
"""

import argparse
import sys
from pathlib import Path

# Runnable straight from a clone, without `pip install -e .` first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarn.capabilities.doc_intelligence import (  # noqa: E402
    ARTIFACTS_DIR,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    DocumentInspector,
    DocumentIntelligenceError,
    create_mock_document,
)

DEMO_ARTIFACT = "annotated_doc_sample.png"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract fields with bounding boxes from a document and annotate it.")
    parser.add_argument("document", nargs="?",
                        help="PDF or image to inspect. Omit to generate a mock invoice.")
    parser.add_argument("--page", type=int, default=1, help="1-based page number (PDFs).")
    parser.add_argument("--backend", choices=("auto", "vlm", "text", "ocr", "mock"),
                        default=None,
                        help="Extraction backend. Default: auto — vlm if a key is set, "
                             "else the PDF's text layer, else local OCR. 'mock' returns "
                             "SYNTHETIC data and never runs on a real document unless "
                             "you name it explicitly.")
    parser.add_argument("--out", default=None,
                        help=f"Annotated image filename, relative to {ARTIFACTS_DIR}. "
                             f"Defaults to <document>_p<page>_annotated.png for a real "
                             f"document, {DEMO_ARTIFACT} for the generated mock.")
    parser.add_argument("--json-out", default=None,
                        help="Also write the full result JSON to this path.")
    args = parser.parse_args(argv)

    document = args.document
    backend = args.backend
    output_path = args.out

    if not document:
        # The synthetic path: we generate the document, so mock extraction
        # describes it correctly and the demo runs with no configuration.
        document = create_mock_document(ARTIFACTS_DIR / "mock_invoice.png")
        backend = backend or "mock"
        output_path = output_path or DEMO_ARTIFACT
        print(f"[demo] no document given — generated a mock invoice at {document}")
    else:
        # A REAL document the user supplied. Never substitute synthetic values
        # for it, and never write them under the demo's sample filename —
        # `annotated_doc_sample.png` names the synthetic invoice, and
        # overwriting it with a real document's overlay would leave two
        # unrelated documents sharing one artifact path. Let the inspector's
        # default naming apply: <document>_p<page>_annotated.png.
        output_path = output_path if args.out else None

    inspector = DocumentInspector(backend=backend)
    try:
        result = inspector.process_document(
            document, page_number=args.page, output_path=output_path)
    except DocumentIntelligenceError as exc:
        # An input problem is the user's to fix, not a stack trace to read.
        print(f"[demo] Error: {exc}", file=sys.stderr)
        return 1

    if result.backend == "mock" and args.document:
        # Belt and braces at the CLI edge too: the only way to reach this is
        # an explicit --backend mock on a real document.
        print("[demo] WARNING: --backend mock was requested for a real document. "
              "The values below are SYNTHETIC SAMPLE DATA and are not this "
              "document's contents.", file=sys.stderr)

    print()
    print("─" * 78)
    print(result.summary())
    print("─" * 78)

    # The point of the confidence tiers: tell the reviewer what to actually look at.
    uncertain = result.low_confidence(CONFIDENCE_HIGH)
    if uncertain:
        print(f"\n{len(uncertain)} field(s) below {CONFIDENCE_HIGH:.2f} — verify these "
              f"against the boxes in the annotated image:")
        for field in sorted(uncertain, key=lambda f: f.confidence):
            flag = "CHECK" if field.confidence >= CONFIDENCE_MEDIUM else "REJECT?"
            print(f"  [{flag:>7}] {field.field_name} = {field.field_value!r} "
                  f"({field.confidence:.2f})")

    print("\nStructured JSON")
    print("─" * 78)
    print(result.to_json())

    if args.json_out:
        Path(args.json_out).write_text(result.to_json(), encoding="utf-8")
        print(f"\n[demo] wrote {args.json_out}")

    print(f"\n[demo] annotated image: {result.annotated_image_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
