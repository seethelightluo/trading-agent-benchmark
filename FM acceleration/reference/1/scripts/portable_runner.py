"""Windows-safe FM launcher for the verified P0/P1 scheduler fingerprint.

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

    # Run the verified scheduler bytes unchanged while
    # routing all FactorMiner subprocesses to the uv interpreter on this host.
    run_pipeline.VENV_PY = Path(sys.executable)
    return run_pipeline


def main() -> int:
    pipeline = load_pipeline()
    return int(pipeline.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
