"""
Tests for agent/data_report.py — the findings report.

The report is the one artifact people forward and quote, so it is trusted more
than terminal output. These tests concentrate on the two properties that matter:
the narrative cannot contradict the evidence, and the sections the narrator
cannot edit must state the truth about what cleaning did.
"""

import os
import re

import numpy as np
import pandas as pd

from agent.data_report import WORKSPACE_DIR, _limitations, write_report
from agent.data_analysis import note_correlation, reset_evidence
from agent.ml.data_pipeline import get_data_pipeline

SITUATION = "You asked which product lines to back. I looked at last year's orders."
TAKEAWAYS = ["North outsells every other region by roughly 40%.",
             "Quantity and revenue move together, as you would expect."]


def _df() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 120
    region = np.array(["north", "south"] * (n // 2))
    qty = rng.integers(1, 20, n).astype(float)
    return pd.DataFrame({
        "region": region,
        "qty": qty,
        "revenue": np.where(region == "north", 900.0, 600.0) + qty * 5 + rng.normal(0, 10, n),
    })


def _load(name="rep_test", df=None):
    get_data_pipeline().datasets[name] = _df() if df is None else df
    return name


def _cleanup(*names):
    pipe = get_data_pipeline()
    for n in names:
        pipe.datasets.pop(n, None)
    for n in names:
        for ext in (".md", ".html"):
            path = os.path.join(WORKSPACE_DIR, f"{n}_report{ext}")
            if os.path.exists(path):
                os.remove(path)


def _read(name, ext):
    with open(os.path.join(WORKSPACE_DIR, f"{name}_report{ext}"), encoding="utf-8") as fh:
        return fh.read()


# ─────────────────────────────────────────────── structure

def test_report_uses_the_four_required_sections_in_order():
    reset_evidence()
    name = _load()
    result = write_report(name, SITUATION, TAKEAWAYS)
    assert not result.startswith("Error:"), result
    expected = ["Background", "Key figures", "Key takeaways", "Methodology", "Appendix"]
    md, page = _read(name, ".md"), _read(name, ".html")
    assert re.findall(r"^## (.+)$", md, re.M) == expected
    assert re.findall(r"<h2>([^<]+)</h2>", page) == expected
    _cleanup(name)


def test_report_carries_the_situation_complication_resolution_story():
    reset_evidence()
    name = _load()
    write_report(name, SITUATION, TAKEAWAYS,
                 complication="Two regions behave so differently that one average hides both.",
                 next_steps=["Split the comparison by product line and re-run it."])
    md = _read(name, ".md")
    assert SITUATION in md                          # situation
    assert "The complication." in md and "hides both" in md
    assert "What I propose next" in md and "Split the comparison" in md
    _cleanup(name)


def test_recommendations_are_labelled_as_judgement_not_measurement():
    """A reader must be able to tell what was measured from what someone concluded."""
    reset_evidence()
    name = _load()
    write_report(name, SITUATION, TAKEAWAYS, next_steps=["Look at product line next."])
    for text in (_read(name, ".md"), _read(name, ".html")):
        assert "judgement, not measurement" in text
    _cleanup(name)


# ─────────────────────────────────────────────── the narrative is checked

def test_report_is_refused_when_the_narrative_contradicts_the_data():
    reset_evidence()
    note_correlation("YEAR", "RATING", -0.02)
    name = _load()
    result = write_report(name, SITUATION,
                          ["Ratings are declining steadily year over year."])
    assert result.startswith("Error:")
    assert "CONTRADICTION" in result
    assert not os.path.exists(os.path.join(WORKSPACE_DIR, f"{name}_report.md")), \
        "a refused report must not be written to disk"
    _cleanup(name)


def test_report_is_written_when_the_narrative_matches_the_data():
    reset_evidence()
    note_correlation("qty", "revenue", 0.68)
    name = _load()
    result = write_report(name, SITUATION,
                          ["Quantity and revenue rise together — a clear relationship."])
    assert not result.startswith("Error:"), result
    _cleanup(name)


def test_situation_and_takeaways_are_required():
    reset_evidence()
    name = _load()
    assert write_report(name, "", TAKEAWAYS).startswith("Error:")
    assert write_report(name, SITUATION, []).startswith("Error:")
    assert write_report("nope", SITUATION, TAKEAWAYS).startswith("Error:")
    _cleanup(name)


# ─────────────────────────────────────────────── the non-editable sections

def test_limitations_say_filled_only_when_values_were_actually_filled():
    """The marker records that a value WAS missing, not that it was filled. If
    the impute step was never approved, saying 'filled in' is a falsehood in the
    one section the narrator cannot edit."""
    df = _df()
    df["revenue_was_missing"] = [1] * 20 + [0] * 100
    df.loc[:19, "revenue"] = None                       # never imputed
    still_blank = _limitations(df, "x")
    assert any("still blank" in n and "not filled in" in n for n in still_blank)
    assert not any("were filled in during cleaning" in n for n in still_blank)

    df2 = _df()
    df2["revenue_was_missing"] = [1] * 20 + [0] * 100    # imputed: no nulls left
    filled = _limitations(df2, "x")
    assert any("were filled in during cleaning" in n for n in filled)


def test_limitations_report_capped_values():
    df = _df()
    df["revenue_was_capped"] = [1] * 6 + [0] * 114
    assert any("rewritten to the edge" in n for n in _limitations(df, "x"))


def test_limitations_are_included_in_both_outputs_and_not_editable():
    reset_evidence()
    df = _df()
    df["revenue_was_missing"] = [1] * 30 + [0] * 90
    name = _load("rep_lim", df)
    write_report(name, SITUATION, TAKEAWAYS)
    for text in (_read(name, ".md"), _read(name, ".html")):
        assert "cannot tell you" in text
        assert "revenue" in text
    _cleanup(name)


# ─────────────────────────────────────────────── the shareable file

def test_html_is_self_contained_and_well_formed():
    reset_evidence()
    name = _load()
    write_report(name, SITUATION, TAKEAWAYS, dashboard_url="http://localhost:8501")
    page = _read(name, ".html")
    assert page.startswith("<!doctype html>")
    for tag in ("main", "table"):
        assert page.count(f"<{tag}") == page.count(f"</{tag}>")
    # nothing loaded from an external host — it must open with no network
    assert not re.search(r"(src|href)=['\"]https?://(?!localhost)", page)
    assert "http://localhost:8501" in page
    _cleanup(name)


def test_report_escapes_html_in_the_narrative():
    reset_evidence()
    name = _load()
    write_report(name, "Context with <script>alert(1)</script> in it", TAKEAWAYS)
    page = _read(name, ".html")
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page
    _cleanup(name)


def test_report_tool_is_registered():
    from agent.runtime.tools import TOOL_REGISTRY
    assert "write_report" in TOOL_REGISTRY
    description = TOOL_REGISTRY["write_report"]["description"]
    assert "Background" in description and "Situation" in description


def test_limitations_do_not_depend_on_a_cleaning_record():
    """A file loaded straight from disk used to produce '0 limitations stated'.

    That made the most confident-looking report the one nobody had checked. The
    limits below are properties of the data, not of the pipeline that touched it.
    """
    import pandas as pd
    from agent.data_report import _limitations
    df = pd.DataFrame({
        "Order_ID": range(1, 41),
        "Revenue": [100.0 + i for i in range(40)],
        "Description": ["Quam numquam iste sunt nemo." for _ in range(40)],
    })
    notes = " ".join(_limitations(df, "never_cleaned"))
    assert "No cost or margin data" in notes
    assert "placeholder text" in notes
    assert "Nothing was cleaned or corrected" in notes


def test_limitations_flag_unresolved_merge_columns():
    import pandas as pd
    from agent.data_report import _limitations
    df = pd.DataFrame({"Occasion_x": ["Holi"] * 20, "Occasion_y": ["Holi"] * 20})
    notes = " ".join(_limitations(df, "merged"))
    assert "merge artefact" in notes


def test_a_single_year_cannot_establish_seasonality():
    import pandas as pd
    from agent.data_report import _limitations
    df = pd.DataFrame({
        "Order_Date": pd.date_range("2023-01-01", "2023-12-29", periods=60),
        "Revenue": [10.0] * 60,
    })
    notes = " ".join(_limitations(df, "one_year"))
    assert "month window only" in notes and "seasonal" in notes


def test_methodology_records_how_derived_columns_were_computed():
    """A report that never says Revenue = Price x Quantity cannot be rebuilt."""
    import pandas as pd
    from agent.data_report import _provenance_lines
    df = pd.DataFrame({"Quantity": [float(q) for q in range(1, 13)],
                       "Price": [10.0 * p for p in range(1, 13)]})
    df["Revenue"] = df["Quantity"] * df["Price"]
    text = " ".join(_provenance_lines(df, "orders"))
    assert "derived column" in text and "Revenue" in text
    assert "Quantity" in text and "Price" in text


def test_methodology_says_a_join_was_not_recorded():
    import pandas as pd
    from agent.data_report import _provenance_lines
    df = pd.DataFrame({"Occasion_x": ["Holi"] * 4, "Occasion_y": ["Holi"] * 4, "v": [1, 2, 3, 4]})
    text = " ".join(_provenance_lines(df, "merged"))
    assert "join" in text and "NOT recorded" in text
