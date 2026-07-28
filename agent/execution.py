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

import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "workspace"))
DEFAULT_TIMEOUT = int(os.environ.get("SWARN_EXEC_TIMEOUT", "300"))
MAX_OUTPUT_CHARS = 50_000
SANDBOX_IMAGE = os.environ.get("SWARN_SANDBOX_IMAGE", "python:3.11-slim")


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
        parts.append(self.output if self.output.strip() else "(no output)")
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

    def _ensure_container(self):
        with self._lock:
            if self._container:
                return
            client = self._docker.from_env()
            name = f"swarn-sandbox-{uuid.uuid4().hex[:8]}"
            print(f"[sandbox] starting container '{name}' ({self.image})…")
            self._container = client.containers.run(
                self.image,
                command="tail -f /dev/null",
                name=name, detach=True, auto_remove=True,
                volumes={self.workspace: {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace",
                mem_limit=self.mem_limit, cpu_count=self.cpu_count,
            )

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


def _docker_available() -> bool:
    try:
        import docker
        docker.from_env().ping()
        return True
    except Exception:  # noqa: BLE001
        return False


def make_backend(workspace: Optional[str] = None) -> ExecutionBackend:
    """Fresh backend (the search engine gives each run its own workspace)."""
    forced = os.environ.get("SWARN_SANDBOX", "").lower()
    if forced == "subprocess":
        return SubprocessBackend(workspace)
    if forced == "docker":
        return DockerBackend(workspace)
    if _docker_available():
        return DockerBackend(workspace)
    return SubprocessBackend(workspace)


_backend: Optional[ExecutionBackend] = None


def get_backend() -> ExecutionBackend:
    """Process-wide default backend (used by the ReAct tools)."""
    global _backend
    if _backend is None:
        _backend = make_backend()
        if _backend.name == "subprocess":
            print("[sandbox] Docker unavailable — using local subprocess backend "
                  "(hard timeouts, no container isolation). Set SWARN_SANDBOX=docker to force Docker.")
    return _backend


def close_backend():
    global _backend
    if _backend:
        _backend.close()
        _backend = None
