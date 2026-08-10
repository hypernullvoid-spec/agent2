"""
The way Claude itself handles spreadsheets.

The file goes to a sandbox computer. The model never sees the rows - it writes
Python, the sandbox runs it, and only the small printed output comes back.

Because of this, file size stops mattering. A 200 MB Excel file works the same
as a 200 KB one.

    pip install anthropic
    export ANTHROPIC_API_KEY="sk-ant-..."

    python3 sandbox.py "/home/spoo/Downloads/movies 2.csv" "Which genre rates highest?"
"""

import mimetypes
import os
import sys

import anthropic

MODEL = "claude-opus-5"
BETAS = ["files-api-2025-04-14"]
TOOL = {"type": "code_execution_20260120", "name": "code_execution"}

# Without this the model sometimes runs print(df) and dumps every row back
# into the context window - the exact problem we are trying to avoid.
RULES = """
You are analysing a data file that has been placed on your sandbox.

Rules:
- Never print a whole dataframe. Print aggregates, summaries, or .head(20) at most.
- Look at the shape first (columns, dtypes, row count) before doing real work.
- Use pandas for the calculations. Do not estimate numbers yourself.
- If something looks wrong in the data - missing values, duplicates, odd
  outliers - say so.
- Finish with a short, plain-English answer for a business user.
"""

client = anthropic.Anthropic()


# ----------------------------------------------------------------------
# STEP 1 - PUT THE FILE ON THE SANDBOX
# ----------------------------------------------------------------------

def upload(path):
    """Upload once. Reuse the returned id for as many questions as you like."""
    media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"

    with open(path, "rb") as f:
        uploaded = client.beta.files.upload(
            file=(os.path.basename(path), f, media_type),
            betas=BETAS,
        )
    return uploaded.id


# ----------------------------------------------------------------------
# STEP 2 - ASK. THE MODEL WRITES AND RUNS THE CODE ITSELF.
# ----------------------------------------------------------------------

def ask(question, file_id=None, container_id=None, history=None):
    """
    Ask a question about the uploaded file.

    Pass container_id from a previous answer to keep the same sandbox alive -
    the file stays loaded and follow-up questions are much faster.
    """
    content = [{"type": "text", "text": f"{RULES}\n\nQuestion: {question}"}]
    if file_id:
        content.append({"type": "container_upload", "file_id": file_id})

    messages = (history or []) + [{"role": "user", "content": content}]

    kwargs = {
        "model": MODEL,
        "max_tokens": 8000,
        "betas": BETAS,
        "messages": messages,
        "tools": [TOOL],
    }
    if container_id:
        kwargs["container"] = container_id

    response = client.beta.messages.create(**kwargs)
    return _unpack(response, messages)


# ----------------------------------------------------------------------
# STEP 3 - READ THE REPLY
# ----------------------------------------------------------------------

def _unpack(response, messages):  # noqa: keeps raw response for save_outputs
    """
    A reply is a list of blocks. Pull out the three things we care about:
    what the model said, what code it ran, and what that code printed.
    """
    said, ran, printed = [], [], []

    for block in response.content:
        kind = getattr(block, "type", "")

        if kind == "text":
            said.append(block.text)

        elif kind == "server_tool_use":
            command = getattr(block, "input", {}) or {}
            ran.append(command.get("command") or command.get("file_text") or str(command))

        elif kind.endswith("_tool_result"):
            result = getattr(block, "content", None)
            out = getattr(result, "stdout", "") or ""
            err = getattr(result, "stderr", "") or ""
            if out.strip():
                printed.append(out.strip())
            if err.strip():
                printed.append(f"[error] {err.strip()}")

    return {
        "answer": "\n".join(said).strip(),
        "code": ran,
        "output": printed,
        "response": response,        # keep the raw reply so save_outputs() can use it
        "container_id": getattr(response.container, "id", None),
        "history": messages + [{"role": "assistant", "content": response.content}],
        "tokens": {
            "in": response.usage.input_tokens,
            "out": response.usage.output_tokens,
        },
    }


# ----------------------------------------------------------------------
# STEP 4 - SAVE ANY CHARTS THE MODEL DREW
# ----------------------------------------------------------------------

def save_outputs(response_obj, folder="charts"):
    """The sandbox can write PNG files. Pull them out and save them locally."""
    os.makedirs(folder, exist_ok=True)
    saved = []

    for block in response_obj.content:
        if not getattr(block, "type", "").endswith("_tool_result"):
            continue
        result = getattr(block, "content", None)
        for item in getattr(result, "content", None) or []:
            file_id = getattr(item, "file_id", None)
            if not file_id:
                continue
            meta = client.beta.files.retrieve_metadata(file_id, betas=BETAS)
            data = client.beta.files.download(file_id, betas=BETAS)
            path = os.path.join(folder, meta.filename)
            data.write_to_file(path)
            saved.append(path)

    return saved


# ----------------------------------------------------------------------
# TRY IT
# ----------------------------------------------------------------------

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/spoo/Downloads/movies 2.csv"
    question = " ".join(sys.argv[2:]) or "Analyse this data and tell me what stands out."

    print(f"Uploading {os.path.basename(path)} ...")
    file_id = upload(path)

    print(f"Asking: {question}\n")
    result = ask(question, file_id=file_id)

    for i, code in enumerate(result["code"], 1):
        print(f"--- code the model ran ({i}) ---")
        print(code[:600])
        print()

    for out in result["output"]:
        print("--- what it printed ---")
        print(out[:800])
        print()

    print("--- answer ---")
    print(result["answer"])
    print()
    print(f"tokens: {result['tokens']['in']:,} in / {result['tokens']['out']:,} out")
    print(f"container: {result['container_id']}   (reuse this for follow-ups)")


if __name__ == "__main__":
    main()
