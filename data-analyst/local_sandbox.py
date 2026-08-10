"""
The sandbox idea, on YOUR server, with ANY model.

The model writes Python. We run it in a locked-down process. Only what the
code printed comes back. Repeat until it has enough to answer.

This is exactly what Claude's code execution tool does - the difference is that
the sandbox is yours, so the data never leaves your machine and any provider
works, including Groq.

    pip install pandas
    export GROQ_API_KEY=...        (or ANTHROPIC_API_KEY, or OPENAI_API_KEY)

    python3 local_sandbox.py "/home/spoo/Downloads/movies 2.csv" "Which genre rates highest?"
"""

import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile

import llm

MAX_ROUNDS = 5           # how many write-run-read cycles before we stop
TIMEOUT_SECONDS = 45
MEMORY_LIMIT_GB = 2
MAX_OUTPUT_CHARS = 4000  # a rogue print(df) must not flood the context window

RULES = """You are a data analyst with a Python sandbox.

The file is at: {path}
Its first lines look like:
{preview}

Write Python to answer the question. pandas, numpy and matplotlib are available.

Rules:
- Reply with ONE ```python code block and nothing else, OR your final answer in
  plain English with NO code block at all.
- STOP AND ANSWER as soon as the output above contains the number you need.
  Do not re-run code that already worked. Repeating a step wastes time and
  changes nothing - the data does not change between rounds.
- NEVER print a whole dataframe. Print aggregates, .describe(), .head(20) at most.
  Output is truncated at {max_chars} characters, so a big dump loses everything.
- Look at the shape first (columns, dtypes, row count) before real work.
- The working folder is KEPT between rounds. Loading a big file every round is
  slow, so on your first step save what you need:
      df.to_parquet("cache.parquet")
  and in later steps read that instead.
- Let pandas do the arithmetic. Never estimate a number yourself.
- Keep the final answer under 150 words, for a business reader.
"""


# ----------------------------------------------------------------------
# THE SANDBOX
# ----------------------------------------------------------------------

def _apply_limits():
    """Runs inside the child process, just before the code does."""
    limit = MEMORY_LIMIT_GB * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))        # memory
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_SECONDS, TIMEOUT_SECONDS))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))              # no core dumps
    # No RLIMIT_NPROC: numpy's BLAS spawns worker threads at import and a low
    # limit makes every pandas script fail. Threads are capped through the
    # environment instead (see SAFE_ENV).


# One thread per library. Keeps CPU use predictable and, more importantly,
# stops numpy failing at import under the memory limit.
SAFE_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "MPLBACKEND": "Agg",          # matplotlib with no screen attached
}


def run_code(code, workdir):
    """
    Run model-written Python in a separate process with hard limits.

    Never use exec() for this. The code came from a language model, so it gets
    its own process, its own directory, a memory cap and a time cap.
    """
    script = os.path.join(workdir, "step.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write(code)

    try:
        finished = subprocess.run(
            [sys.executable, script],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            preexec_fn=_apply_limits,
            env={"PATH": os.environ.get("PATH", ""), "HOME": workdir, **SAFE_ENV},
        )
        out, err, ok = finished.stdout, finished.stderr, finished.returncode == 0
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": "", "error": f"Timed out after {TIMEOUT_SECONDS}s."}

    if len(out) > MAX_OUTPUT_CHARS:
        out = out[:MAX_OUTPUT_CHARS] + f"\n... [cut - printed more than {MAX_OUTPUT_CHARS} characters]"

    return {"ok": ok, "output": out.strip(), "error": err.strip()[-1500:]}


SANDBOX_IMAGE = "data-analyst-sandbox:v2"   # bump the tag to force a rebuild

DOCKERFILE = """FROM python:3.11-slim
RUN pip install --no-cache-dir pandas numpy matplotlib openpyxl scipy \
    pyarrow duckdb
RUN useradd -m runner
USER runner
"""


def docker_available():
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=10, check=True)
        return True
    except Exception:
        return False


def ensure_image(quiet=False):
    """
    Build the sandbox image once, the first time it is needed.

    It has to be built rather than pulled because the container runs with
    --network=none, so it cannot pip install anything at run time.
    """
    exists = subprocess.run(["docker", "image", "inspect", SANDBOX_IMAGE],
                            capture_output=True)
    if exists.returncode == 0:
        return True

    if not quiet:
        print(f"Building {SANDBOX_IMAGE} (one time, a few minutes)...")

    build = subprocess.run(["docker", "build", "-t", SANDBOX_IMAGE, "-"],
                           input=DOCKERFILE, text=True,
                           capture_output=True, timeout=900)
    if build.returncode != 0 and not quiet:
        print("Build failed:", build.stderr[-400:])
    return build.returncode == 0


def run_code_docker(code, workdir, image=SANDBOX_IMAGE):
    """
    Stronger isolation, if you have Docker. --network=none is the real win:
    the code cannot reach the internet or anything else on your network.
    """
    script = os.path.join(workdir, "step.py")
    with open(script, "w", encoding="utf-8") as f:
        f.write(code)

    try:
        finished = subprocess.run(
            ["docker", "run", "--rm",
             "--network=none",                    # the real isolation win
             "--memory", f"{MEMORY_LIMIT_GB}g", "--cpus", "1",
             "--pids-limit", "128",
             "-v", f"{workdir}:/work", "-w", "/work",
             "--user", f"{os.getuid()}:{os.getgid()}",
             image, "python", "step.py"],
            capture_output=True, text=True, timeout=TIMEOUT_SECONDS + 20,
        )
        out, err, ok = finished.stdout, finished.stderr, finished.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError) as error:
        return {"ok": False, "output": "", "error": str(error)}

    if len(out) > MAX_OUTPUT_CHARS:
        out = out[:MAX_OUTPUT_CHARS] + "\n... [cut]"
    return {"ok": ok, "output": out.strip(), "error": err.strip()[-1500:]}


# ----------------------------------------------------------------------
# THE LOOP
# ----------------------------------------------------------------------

def extract_code(reply):
    """Pull the python block out of the reply. No block means it is answering."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", reply, re.DOTALL)
    return match.group(1).strip() if match else None


def _same_code(a, b):
    """
    Same code ignoring comments, blank lines and ALL spacing.

    The model rewrites the same step with different comments and spacing, so a
    plain string compare misses it. Normalise both before comparing.
    """
    def strip(text):
        lines = [" ".join(line.split("#")[0].split()) for line in text.splitlines()]
        return " ".join(line for line in lines if line)
    return strip(a) == strip(b)


def preview(path, lines=3):
    """A couple of raw lines, so the model knows the shape before it writes code."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "\n".join(next(f).rstrip()[:200] for _ in range(lines))
    except Exception:
        return "(binary file - use pandas.read_excel)"


def analyse(path, question, rounds=MAX_ROUNDS, use_docker=None, quiet=False):
    """Write code, run it, read the output, repeat. Then answer."""
    # Prefer Docker: a plain subprocess can still reach the network, Docker
    # with --network=none cannot.
    if use_docker is None:
        use_docker = docker_available() and ensure_image(quiet)
    if not use_docker and not quiet:
        print("WARNING: running without Docker. The sandboxed code CAN reach "
              "the network. Fine for your own files, not for untrusted uploads.")

    workdir = tempfile.mkdtemp(prefix="analysis_")
    runner = run_code_docker if use_docker else run_code

    try:
        # Only the one data file goes into the sandbox directory
        local_name = os.path.basename(path)
        shutil.copy(path, os.path.join(workdir, local_name))

        conversation = RULES.format(path=local_name, preview=preview(path),
                                    max_chars=MAX_OUTPUT_CHARS)
        conversation += f"\n\nQuestion: {question}\n"
        steps = []

        for step in range(1, rounds + 1):
            reply = llm.chat(conversation, tier="large", max_tokens=1500)
            code = extract_code(reply)

            if not code:                                   # no code = final answer
                return {"answer": reply, "steps": steps}

            # A model that re-runs code it already ran is stuck in a loop. It
            # has the answer and does not realise it. Stop and make it answer.
            if any(_same_code(code, done["code"]) for done in steps):
                if not quiet:
                    print(f"--- round {step}: same code as before, stopping ---\n")
                break

            if not quiet:
                print(f"--- round {step}: code ---")
                print(code[:500])

            result = runner(code, workdir)
            shown = result["output"] or result["error"] or "(printed nothing)"

            if not quiet:
                print(f"--- round {step}: output ---")
                print(shown[:600])
                print()

            steps.append({"code": code, "result": shown})
            conversation += f"\n```python\n{code}\n```\n\nOutput:\n{shown}\n"

            if not result["ok"]:
                conversation += "\nThat failed. Fix it and try again.\n"

        # Either the model looped, or we ran out of rounds. Either way, make it
        # answer with what it already has.
        conversation += ("\nStop writing code. You already have what you need. "
                         "Give your final answer in plain English now, no code.")
        return {"answer": llm.chat(conversation, tier="large", max_tokens=800),
                "steps": steps}

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    args = sys.argv[1:]
    path = args[0] if args and os.path.exists(args[0]) else "/home/spoo/Downloads/movies 2.csv"
    question = " ".join(args[1:] if args and os.path.exists(args[0]) else args) \
        or "Analyse this data and tell me what stands out."

    print(f"Provider: {llm.info()}")
    print(f"File    : {os.path.basename(path)}")
    result = analyse(path, question)
    print("--- answer ---")
    print(result["answer"])
    print(f"\n({len(result['steps'])} rounds of code)")


if __name__ == "__main__":
    main()
