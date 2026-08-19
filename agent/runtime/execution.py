"""
Execution backends — run agent-generated code with or without Docker.

  DockerBackend      one persistent container per session, workspace bind-
                     mounted, memory/CPU capped (the Phase 2 design, kept).
  SubprocessBackend  cross-platform local execution with hard wall-clock
                     timeouts. Uses sys.executable (works on Windows, where
                     the old fallback's hardcoded "python3" broke).

get_backend() auto-detects: Docker if the daemon responds, else subprocess.
Force one with SWARN_SANDBOX=docker|subprocess.

Both backends implement the same three methods, and both accept a per-call
`timeout` override — ML training runs need minutes-to-hours, a quick data
peek needs seconds; one global constant can't serve both.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from agent.paths import WORKSPACE_DIR
from agent.config import DEFAULT_TIMEOUT, SANDBOX_IMAGE, sandbox_mode
MAX_OUTPUT_CHARS = 50_000

# The default image is a bare Python — it has no data-science stack, so every
# generated script dies on `import pandas`. Installed once per container.
# Set SWARN_SANDBOX_PACKAGES="" to skip, or point SWARN_SANDBOX_IMAGE at an
# image that already ships them.
SANDBOX_PACKAGES = os.environ.get(
    "SWARN_SANDBOX_PACKAGES", "pandas numpy scipy scikit-learn matplotlib openpyxl")
SANDBOX_INSTALL_TIMEOUT = int(os.environ.get("SWARN_SANDBOX_INSTALL_TIMEOUT", "900"))

# Once the stack is installed the container is committed to this image, so the
# minute-long pip install happens once ever rather than once per container —
# a parallel search spawns one backend per run and would otherwise pay it N times.
SANDBOX_READY_IMAGE = os.environ.get("SWARN_SANDBOX_READY_IMAGE", "swarn-sandbox:ready")

# pip name → import name, where they differ
_IMPORT_NAMES = {"scikit-learn": "sklearn", "pillow": "PIL", "beautifulsoup4": "bs4",
                 "opencv-python": "cv2", "python-dateutil": "dateutil"}


@dataclass
class ExecResult:
    """Structured result — the search engine needs exit codes and timing,
    not just a display string."""
    output: str
    exit_code: int = 0
    timed_out: bool = False
    exec_time: float = 0.0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def as_text(self) -> str:
        """Legacy display format used by the ReAct tools."""
        parts = []
        if self.timed_out:
            parts.append(f"Error: command timed out after {self.exec_time:.0f}s")
        elif self.exit_code != 0:
            parts.append(f"[exit {self.exit_code}]")
        if self.output.strip():
            parts.append(self.output)
        elif self.ok:
            # bare "(no output)" reads like a failure, so the model retries or
            # gives up when in fact its code simply never printed anything
            parts.append("(the code ran successfully, exit 0, but printed nothing — "
                         "Python scripts do not echo values like a notebook. "
                         "Add print(...) around what you want to see.)")
        else:
            parts.append("(no output)")
        return "\n".join(parts)


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n… [{len(text) - limit} chars truncated] …\n" + text[-half:]


# ─────────────────────────────────────────────────────── subprocess backend

class SubprocessBackend:
    """Local execution with wall-clock timeout. Not isolated — documented,
    deliberate trade-off so the agent works on any machine without Docker."""

    name = "subprocess"

    def __init__(self, workspace: Optional[str] = None):
        self.workspace = os.path.abspath(workspace or WORKSPACE_DIR)
        os.makedirs(self.workspace, exist_ok=True)

    def exec_python(self, code: str, timeout: Optional[int] = None) -> ExecResult:
        script = os.path.join(self.workspace, f"_exec_{uuid.uuid4().hex[:8]}.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            return self._run([sys.executable, script], timeout)
        finally:
            try:
                os.remove(script)
            except OSError:
                pass

    def exec_shell(self, command: str, timeout: Optional[int] = None) -> ExecResult:
        if os.name == "nt":
            cmd = ["cmd", "/c", command]
        else:
            cmd = ["bash", "-c", command]
        return self._run(cmd, timeout)

    def _run(self, cmd: list[str], timeout: Optional[int]) -> ExecResult:
        timeout = timeout or DEFAULT_TIMEOUT
        start = time.time()
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        try:
            r = subprocess.run(
                cmd, cwd=self.workspace, capture_output=True, text=True,
                timeout=timeout, env=env, errors="replace",
            )
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or "") + (("\n[stderr]\n" + e.stderr) if e.stderr else "")
            if isinstance(out, bytes):
                out = out.decode("utf-8", errors="replace")
            return ExecResult(output=_truncate(str(out)), exit_code=-1,
                              timed_out=True, exec_time=time.time() - start)
        out = r.stdout or ""
        if r.stderr:
            out += ("\n[stderr]\n" if out else "[stderr]\n") + r.stderr
        return ExecResult(output=_truncate(out.strip()), exit_code=r.returncode,
                          exec_time=time.time() - start)

    def close(self):
        pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # exists, owned by someone else
    return True


def _reap_orphan_containers(client) -> None:
    """Remove sandbox containers whose creating process is gone.

    A container runs `tail -f /dev/null` forever, so a session that crashes or
    is killed before close() leaves it holding memory indefinitely. Matching on
    the recorded PID means a concurrently running session is never touched.
    """
    try:
        stale = client.containers.list(filters={"label": "swarn.sandbox=1"})
    except Exception:  # noqa: BLE001
        return
    for container in stale:
        pid = container.labels.get("swarn.pid")
        if not pid or not pid.isdigit() or _pid_alive(int(pid)):
            continue
        try:
            container.remove(force=True)
            print(f"[sandbox] reaped orphaned container '{container.name}' "
                  f"(its session, pid {pid}, is gone)")
        except Exception:  # noqa: BLE001
            pass


# ─────────────────────────────────────────────────────────── docker backend

class DockerBackend:
    """One persistent container per session; workspace bind-mounted at
    /workspace. Falls back never — get_backend() only returns this when the
    daemon already answered a ping."""

    name = "docker"

    def __init__(self, workspace: Optional[str] = None, image: str = SANDBOX_IMAGE,
                 mem_limit: str = "2g", cpu_count: int = 2):
        import docker
        self._docker = docker
        self.workspace = os.path.abspath(workspace or WORKSPACE_DIR)
        os.makedirs(self.workspace, exist_ok=True)
        self.image = image
        self.mem_limit = mem_limit
        self.cpu_count = cpu_count
        self._container = None
        self._lock = threading.Lock()
        self._bootstrapped = False

    def _ensure_container(self):
        with self._lock:
            if self._container:
                return
            client = self._docker.from_env()
            image = self.image
            if SANDBOX_PACKAGES.strip() and self.image == SANDBOX_IMAGE:
                try:                            # reuse the prepared image if we built one
                    client.images.get(SANDBOX_READY_IMAGE)
                    image = SANDBOX_READY_IMAGE
                except Exception:               # noqa: BLE001 — not built yet
                    pass
            _reap_orphan_containers(client)
            name = f"swarn-sandbox-{uuid.uuid4().hex[:8]}"
            print(f"[sandbox] starting container '{name}' ({image})…")
            self._container = client.containers.run(
                image,
                command="tail -f /dev/null",
                name=name, detach=True, auto_remove=True,
                volumes={self.workspace: {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace",
                mem_limit=self.mem_limit, cpu_count=self.cpu_count,
                labels={"swarn.sandbox": "1", "swarn.pid": str(os.getpid())},
            )
            # The prepared image carries whatever SANDBOX_PACKAGES said WHEN IT WAS
            # BUILT. Trusting it blindly means a package added to the list later is
            # silently ignored until someone thinks to delete the cached image —
            # the failure then surfaces as ImportError inside generated code, far
            # from its cause. The probe is one cheap exec, so it always runs; only
            # the minute-long install is skipped when nothing is actually missing.
            self._bootstrap_packages()

    def _bootstrap_packages(self) -> None:
        """Install the data-science stack into a fresh container, once.

        Only what is actually missing is installed, so a richer
        SWARN_SANDBOX_IMAGE costs nothing. Failures are reported rather than
        raised — the sandbox still works for code that needs no extras.
        """
        if self._bootstrapped:
            return
        self._bootstrapped = True
        wanted = [p for p in SANDBOX_PACKAGES.split() if p]
        if not wanted:
            return
        modules = {p: _IMPORT_NAMES.get(p, p.replace("-", "_")) for p in wanted}
        probe = (
            "import importlib.util, json\n"
            f"mods = json.loads({json.dumps(json.dumps(modules))})\n"
            "print(json.dumps([p for p, m in mods.items() "
            "if importlib.util.find_spec(m) is None]))"
        )
        missing = wanted
        result = self._run(["python3", "-c", probe], timeout=120)
        for line in result.output.splitlines():
            line = line.strip()
            if line.startswith("["):
                try:
                    missing = json.loads(line)
                    break
                except json.JSONDecodeError:
                    pass
        if not missing:
            return
        print(f"[sandbox] installing {' '.join(missing)} into the container — "
              f"one-time, may take a minute…")
        install = self._run(
            ["pip", "install", "--no-cache-dir", "--disable-pip-version-check", "-q"] + missing,
            timeout=SANDBOX_INSTALL_TIMEOUT)
        if install.ok:
            print(f"[sandbox] ready ({len(missing)} package(s) installed).")
            repo, _, tag = SANDBOX_READY_IMAGE.partition(":")
            try:
                self._container.commit(repository=repo, tag=tag or "latest")
                print(f"[sandbox] cached as {SANDBOX_READY_IMAGE} — later containers start "
                      f"instantly. Delete that image to rebuild it.")
            except Exception as e:  # noqa: BLE001 — caching is an optimisation, not a requirement
                print(f"[sandbox] (could not cache the prepared image: {type(e).__name__}: {e})")
        else:
            print(f"[sandbox] WARNING: could not install {missing} — generated code that "
                  f"imports them will fail. Set SWARN_SANDBOX_PACKAGES='' to skip this step, "
                  f"or SWARN_SANDBOX_IMAGE to an image that already has them.\n"
                  f"          {install.output.strip()[:300]}")

    def exec_python(self, code: str, timeout: Optional[int] = None) -> ExecResult:
        self._ensure_container()
        script_name = f"_exec_{uuid.uuid4().hex[:8]}.py"
        with open(os.path.join(self.workspace, script_name), "w", encoding="utf-8") as f:
            f.write(code)
        try:
            return self._run(["python3", f"/workspace/{script_name}"], timeout)
        finally:
            try:
                os.remove(os.path.join(self.workspace, script_name))
            except OSError:
                pass

    def exec_shell(self, command: str, timeout: Optional[int] = None) -> ExecResult:
        self._ensure_container()
        return self._run(["bash", "-c", command], timeout)

    def _run(self, cmd: list[str], timeout: Optional[int]) -> ExecResult:
        timeout = timeout or DEFAULT_TIMEOUT
        start = time.time()
        slot: dict = {"done": False, "result": None, "error": None}

        def worker():
            try:
                exit_code, (out_b, err_b) = self._container.exec_run(
                    cmd=cmd, workdir="/workspace", demux=True)
                slot["result"] = (exit_code, out_b, err_b)
            except Exception as exc:  # noqa: BLE001
                slot["error"] = str(exc)
            finally:
                slot["done"] = True

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout=timeout)
        elapsed = time.time() - start

        if not slot["done"]:
            # V3 fix: abandoning the worker thread used to leave the command
            # running inside the container, silently eating CPU/RAM and
            # skewing every later run. Docker can't kill a single exec, so
            # kill the whole container and recreate it lazily on next use.
            self._recycle_container()
            return ExecResult(output="", exit_code=-1, timed_out=True, exec_time=elapsed)
        if slot["error"]:
            return ExecResult(output=f"Error executing in sandbox: {slot['error']}",
                              exit_code=1, exec_time=elapsed)
        exit_code, out_b, err_b = slot["result"]
        out = (out_b or b"").decode("utf-8", errors="replace").strip()
        err = (err_b or b"").decode("utf-8", errors="replace").strip()
        if err:
            out += ("\n[stderr]\n" if out else "[stderr]\n") + err
        return ExecResult(output=_truncate(out), exit_code=exit_code or 0, exec_time=elapsed)

    def _recycle_container(self):
        """Kill a container whose exec timed out; the next call recreates it."""
        with self._lock:
            if not self._container:
                return
            try:
                print(f"[sandbox] timeout — killing container {self._container.short_id} "
                      "to stop the runaway process…")
                self._container.kill()
            except Exception:  # noqa: BLE001
                pass
            self._container = None  # auto_remove=True cleans up the husk

    def close(self):
        if self._container:
            try:
                print(f"[sandbox] stopping container {self._container.short_id}…")
                self._container.stop(timeout=5)
            except Exception:  # noqa: BLE001
                pass
            self._container = None


# ──────────────────────────────────────────────────────────────── selection

ExecutionBackend = SubprocessBackend | DockerBackend


def _docker_check() -> tuple[bool, str]:
    """(available, reason-if-not). Distinguishes 'the docker Python package is
    missing' from 'the daemon is down' — reporting both as "Docker unavailable"
    sends people to check a daemon that is running perfectly well.
    """
    try:
        import docker
    except ImportError:
        return False, ("the 'docker' Python package is not installed "
                       "(the Docker daemon itself may be fine) — run: pip install docker")
    try:
        docker.from_env().ping()
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, (f"cannot reach the Docker daemon ({type(e).__name__}) — "
                       "is Docker running, and is your user in the 'docker' group?")


def _docker_available() -> bool:
    return _docker_check()[0]


def make_backend(workspace: Optional[str] = None) -> ExecutionBackend:
    """Fresh backend (the search engine gives each run its own workspace)."""
    forced = sandbox_mode()
    if forced == "subprocess":
        return SubprocessBackend(workspace)
    if forced == "docker":
        return DockerBackend(workspace)
    available, reason = _docker_check()
    if available:
        return DockerBackend(workspace)
    backend = SubprocessBackend(workspace)
    backend.unavailable_reason = reason
    return backend


_backend: Optional[ExecutionBackend] = None


def get_backend() -> ExecutionBackend:
    """Process-wide default backend (used by the ReAct tools)."""
    global _backend
    if _backend is None:
        _backend = make_backend()
        if _backend.name == "subprocess":
            reason = getattr(_backend, "unavailable_reason", "")
            detail = f" — {reason}" if reason else ""
            print(f"[sandbox] Falling back to the local subprocess backend{detail}\n"
                  "          Your code runs directly on this machine, with hard timeouts "
                  "but no container isolation.")
    return _backend


def close_backend():
    global _backend
    if _backend:
        _backend.close()
        _backend = None


@atexit.register
def _close_backend_at_exit():
    """A sandbox container outlives its Python process unless told to stop, so a
    session that simply ends would leak one. Best-effort: never raise on exit."""
    try:
        close_backend()
    except Exception:  # noqa: BLE001
        pass
