"""
The web API - now running the same pipeline as the CLI.

    uvicorn app:app --reload
    open http://localhost:8000/docs

    POST /upload   send a CSV or Excel file  -> file_id + what was detected
    POST /ask      send a question           -> answer + how it was worked out

Works with whichever API key is set - see llm.py.
"""

import os
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

import analyst
import llm
import loader

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Data Analyst")

# file_id -> {"name", "tables": {table: rows}, "prep": {table: [notes]}}
# In-memory, so uploads are lost on restart. Move to Redis or store.py
# (Parquet + DuckDB) before this carries real traffic.
FILES = {}


class Question(BaseModel):
    file_id: str
    question: str
    table: str | None = None      # which sheet, if the Excel file has several


# ----------------------------------------------------------------------
# UPLOAD - read once, prepare once
# ----------------------------------------------------------------------

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(400, "Please upload a CSV or Excel file.")

    file_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    with open(path, "wb") as f:
        f.write(await file.read())

    try:
        raw_tables = loader.read_file(path)
    except Exception as error:
        raise HTTPException(400, f"Could not read that file: {error}")

    tables, prep, summary = {}, {}, []

    for name, rows in raw_tables.items():
        rows = loader.clean(rows)
        if not rows:
            continue

        # Split multi-value cells, tidy messy year columns. Automatic - nothing
        # here is specific to any one file.
        rows, notes = loader.auto_prepare(rows)
        dimensions, measures = loader.classify_columns(rows)

        tables[name] = rows
        prep[name] = notes
        summary.append({
            "name": name,
            "rows": len(rows),
            "columns": list(rows[0]),
            "can_group_by": dimensions,
            "can_measure": measures,
            "prepared": notes,          # tell the user what we changed
        })

    if not tables:
        raise HTTPException(400, "That file has no readable data in it.")

    FILES[file_id] = {"name": file.filename, "tables": tables, "prep": prep}
    return {"file_id": file_id, "provider": llm.provider(), "tables": summary}


# ----------------------------------------------------------------------
# ASK - group for the question, then answer
# ----------------------------------------------------------------------

@app.post("/ask")
def ask(body: Question):
    stored = FILES.get(body.file_id)
    if not stored:
        raise HTTPException(404, "Unknown file_id. Upload the file again.")

    table = body.table or next(iter(stored["tables"]))
    rows = stored["tables"].get(table)
    if not rows:
        raise HTTPException(404, f"No table called '{table}'.")

    dimensions, measures = loader.classify_columns(rows)

    # Without any groupable column there is nothing to roll up, so fall back to
    # sending rows directly and let fit_to_budget trim them.
    if not dimensions or not measures:
        result = analyst.ask(body.question, rows, quiet=True)
        return {"table": table, "grouped_by": None,
                "prepared": stored["prep"].get(table, []), **result}

    # Which columns should this question be grouped by?
    groups = analyst.pick_grouping(body.question, dimensions)

    # A time trend must not mix kinds of record - see loader.title_type.
    for dim in dimensions:
        if dim.endswith("_TYPE") and dim not in groups:
            base = dim[:-5]
            if base in groups:
                groups = groups + [dim]

    summary = loader.rollup(
        rows, groups,
        {m: "avg" for m in measures[:3]},      # keep the payload small
        min_rows=20,
    )

    if not summary:                            # every group was too small
        summary = loader.rollup(rows, groups, {m: "avg" for m in measures[:3]})

    caveats = loader.reliability_notes(summary, groups)

    result = analyst.ask(body.question, summary, caveats=caveats,
                         group_by=groups, quiet=True)

    return {
        "table": table,
        "grouped_by": groups,
        "summary_rows": len(summary),
        "prepared": stored["prep"].get(table, []),
        "caveats": caveats,
        **result,
    }


@app.get("/health")
def health():
    return {"ok": True, "provider": llm.info(), "files_loaded": len(FILES)}
