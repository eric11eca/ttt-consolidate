"""Grade math answers in a long-lived worker process with a hard timeout.

Why a separate process:
* sympy can run for minutes on inputs like ``\\boxed{9^{9^{9^{9}}}}`` and
  SIGALRM cannot interrupt it; a worker can be killed.
* sympy's LaTeX parser needs antlr4 4.11 while hydra/omegaconf need 4.9. The
  worker puts the vendored 4.11 runtime first on sys.path and never imports
  omegaconf, so the main process keeps its own antlr4 untouched.

The worker is a fresh ``python -m opt_math.grading.pool`` subprocess (not a
multiprocessing child, so the caller's ``__main__`` is never re-imported). It
talks JSON lines over stdin/stdout, is started on first use and reused. Each
request carries an id that the reply echoes; a reply that does not parse or
carries the wrong id kills the worker, which is restarted for one retry. On a
timeout the worker is killed and restarted, and the answer counts as wrong. If
the LaTeX backend cannot load, ``LatexBackendError`` is raised to the caller.
"""

import atexit
import json
import logging
import os
import select
import subprocess
import sys
import tempfile
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

GRADE_TIMEOUT_S = 20
STARTUP_TIMEOUT_S = 180


def _worker_main() -> None:
    """Worker loop. Runs in the subprocess."""
    # Keep the real stdout for the protocol only; anything else that writes to
    # fd 1 (C code, child processes, sitecustomize) lands on stderr instead.
    protocol_out = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    logging.disable(logging.ERROR)

    def reply(kind, payload=None, req_id=None):
        protocol_out.write(json.dumps([kind, payload, req_id]) + "\n")
        protocol_out.flush()

    from .third_party.matharena.parser import LatexBackendError, ensure_latex_backend, prepare_vendored_antlr

    prepare_vendored_antlr()
    try:
        ensure_latex_backend()
    except LatexBackendError as exc:
        reply("backend_error", str(exc))
        return
    from .patches import grade_core

    reply("ready")
    for line in sys.stdin:
        req_id, text, gold, source = json.loads(line)
        try:
            reply("ok", list(grade_core(text, gold, source)), req_id)
        except LatexBackendError as exc:
            reply("backend_error", str(exc), req_id)
        except BaseException:  # never let one answer kill the worker
            reply("ok", [False, ""], req_id)


class _Desync(Exception):
    pass


class _Timeout(Exception):
    pass


class _Eof(Exception):
    """The worker closed its output (it exited or was killed)."""


class MathGradingWorker:
    def __init__(self, timeout_s: float = GRADE_TIMEOUT_S):
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._err = None
        self._buf = b""
        self._next_id = 0

    def _read_message(self, timeout_s: float):
        """Return the next decoded message. Raises _Timeout, _Eof or _Desync."""
        deadline = time.monotonic() + timeout_s
        fd = self._proc.stdout.fileno()
        while b"\n" not in self._buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _Timeout()
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                raise _Timeout()
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                raise _Eof()
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        try:
            msg = json.loads(line.decode("utf-8"))
            if not (isinstance(msg, list) and len(msg) == 3):
                raise ValueError("bad message shape")
        except ValueError as exc:  # includes JSONDecodeError and UnicodeDecodeError
            raise _Desync(str(exc)) from exc
        return msg

    def _stderr_tail(self) -> str:
        try:
            self._err.seek(0)
            return self._err.read().decode("utf-8", "replace")[-1500:]
        except Exception:
            return ""

    def _start(self) -> None:
        from .third_party.matharena.parser import LatexBackendError

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in (repo_root, env.get("PYTHONPATH", "")) if p)
        self._buf = b""
        self._err = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "opt_math.grading.pool"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._err,
            env=env,
            cwd=repo_root,
        )
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        msg = None
        while True:
            try:
                msg = self._read_message(max(0.0, deadline - time.monotonic()))
            except _Desync:
                continue  # noise printed during interpreter startup
            except (_Timeout, _Eof):
                msg = None
            break
        if msg is None:
            tail = self._stderr_tail()
            self._kill()
            raise LatexBackendError(f"math grading worker did not start. stderr tail: {tail}")
        kind, payload, _ = msg
        if kind != "ready":
            self._kill()
            raise LatexBackendError(payload)

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        for stream in (proc.stdin, proc.stdout):
            try:
                stream.close()
            except OSError:
                pass
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        if self._err is not None:
            self._err.close()
            self._err = None

    def _roundtrip(self, text: str, gold: str, source: Optional[str]):
        """Send one request on a live worker. Returns a message; raises _Timeout, _Eof or _Desync."""
        self._next_id += 1
        req_id = self._next_id
        self._proc.stdin.write((json.dumps([req_id, text, gold, source]) + "\n").encode("utf-8"))
        self._proc.stdin.flush()
        msg = self._read_message(self.timeout_s)
        if msg[2] != req_id:
            raise _Desync(f"reply id {msg[2]} != request id {req_id}")
        return msg

    def grade(self, text: str, gold: str, source: Optional[str] = None) -> tuple:
        """Return ``(is_correct, extracted, status)`` with status "ok", "timeout" or "error"."""
        from .third_party.matharena.parser import LatexBackendError

        with self._lock:
            for attempt in range(2):
                if self._proc is None or self._proc.poll() is not None:
                    self._kill()
                    self._start()
                try:
                    msg = self._roundtrip(text, gold, source)
                except _Desync as exc:
                    logger.warning("math grading worker out of sync (%s); restarting", exc)
                    self._kill()
                    continue
                except _Timeout:
                    logger.warning("math grading timed out after %ss; restarting worker", self.timeout_s)
                    self._kill()
                    return False, "", "timeout"
                except (_Eof, OSError, BrokenPipeError):
                    logger.warning("math grading worker exited; stderr tail: %s", self._stderr_tail())
                    self._kill()  # the worker died: retry once on a fresh one
                    continue
                kind, payload, _ = msg
                if kind == "backend_error":
                    raise LatexBackendError(payload)
                return bool(payload[0]), str(payload[1]), "ok"
            return False, "", "error"

    def close(self) -> None:
        with self._lock:
            self._kill()


_worker: Optional[MathGradingWorker] = None
_worker_pid: Optional[int] = None
_worker_guard = threading.Lock()


def grade_in_worker(text: str, gold: str, source: Optional[str] = None) -> tuple:
    """Return ``(is_correct, extracted_answer, status)`` using this process's worker."""
    global _worker, _worker_pid
    with _worker_guard:
        # A forked child must not share its parent's pipes: start its own worker.
        if _worker is None or _worker_pid != os.getpid():
            _worker = MathGradingWorker()
            _worker_pid = os.getpid()
            atexit.register(_worker.close)
    return _worker.grade(text, gold, source)


if __name__ == "__main__":
    _worker_main()
