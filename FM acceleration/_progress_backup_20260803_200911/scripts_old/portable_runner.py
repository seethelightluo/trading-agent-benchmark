"""Windows-safe FM launcher that preserves the verified scheduler fingerprint.

The source scheduler retains its original Linux VENV_PY constant because that
file is part of the shared warm-up fingerprint.  This wrapper overrides only
the in-memory child interpreter path after import.  On Windows it also provides
the FM-only no-op fcntl module needed for importing a POSIX-oriented scheduler;
AC mode is deliberately not supported by this bundle.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_FRAMEWORK = ROOT / "bundle" / "agent-framework"
FACTOR_MINER = AGENT_FRAMEWORK / "FactorMiner"


def _install_windows_fcntl_shim() -> None:
    if os.name != "nt" or "fcntl" in sys.modules:
        return
    shim = types.ModuleType("fcntl")
    shim.LOCK_EX = 2
    shim.LOCK_UN = 8

    def flock(_fd: int, _operation: int) -> None:
        # FM-only portable bundle: AC's shared-warmup lock is never entered.
        # Each worker has an independent FM state file, so no FM lock is needed.
        return None

    shim.flock = flock
    sys.modules["fcntl"] = shim


def load_pipeline():
    _install_windows_fcntl_shim()
    sys.path.insert(0, str(AGENT_FRAMEWORK))
    sys.path.insert(0, str(FACTOR_MINER))
    os.chdir(AGENT_FRAMEWORK)
    from scheduler import run_pipeline

    # Keep run_pipeline.py bytes untouched for warm-up fingerprint reuse while
    # routing all FactorMiner subprocesses to the uv interpreter on this host.
    run_pipeline.VENV_PY = Path(sys.executable)

    # The shared warm-up fingerprint digests research-code file paths via
    # `_sha256_paths`, which uses os-native separators. On Windows that yields
    # backslashes and breaks cross-platform warm-up reuse even though the code
    # bytes are identical. Override the in-memory function to normalise path
    # separators to POSIX "/" WITHOUT editing run_pipeline.py (whose bytes are
    # themselves part of the fingerprint). Internal call sites in run_pipeline
    # resolve _sha256_paths through the module global at call time, so this
    # override is honoured by both the verifier and the live pipeline.
    _orig_sha256_paths = run_pipeline._sha256_paths

    def _posix_sha256_paths(paths):
        import hashlib

        here = run_pipeline.HERE
        digest = hashlib.sha256()
        for path in sorted(paths):
            rel = str(path.relative_to(here)).replace("\\", "/")
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    run_pipeline._sha256_paths = _posix_sha256_paths
    return run_pipeline


def main() -> int:
    pipeline = load_pipeline()
    return int(pipeline.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
