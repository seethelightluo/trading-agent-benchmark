#!/usr/bin/env python3
"""Push Git checkpoints when the active DeepSeek AC reaches major milestones."""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path


REPO = Path(os.environ.get("GIT_AUTOPUSH_REPO", "/home/lxx/trade-agent-benchmark"))
STATE = Path(
    os.environ.get(
        "AC_MILESTONE_STATE",
        str(REPO / "AC-deepseek" / "results" / "ac9wl_deepseek" / "run_state.json"),
    )
)
MARKER = REPO / ".git" / "ac_milestone_watch.json"
INTERVAL = max(5, int(os.environ.get("AC_MILESTONE_POLL_SECONDS", "30")))
PUSH = REPO / "ops" / "git_milestone_push.sh"
WL_RE = re.compile(r"^wl([0-9]+)$")


def read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def load_marker() -> dict:
    value = read_json(MARKER)
    return value if isinstance(value, dict) else {}


def save_marker(value: dict) -> None:
    temporary = MARKER.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(MARKER)


def milestones(state: dict) -> dict[str, str]:
    found: dict[str, str] = {}
    warmup = state.get("shared_warmup")
    if isinstance(warmup, dict) and warmup.get("status") == "complete":
        found["deepseek-warmup"] = str(warmup.get("completed_at", ""))

    for key, value in state.items():
        match = WL_RE.match(str(key))
        if not match or not isinstance(value, dict):
            continue
        if value.get("status") == "complete" and value.get("ac_done"):
            found[f"deepseek-wl{match.group(1)}"] = str(
                value.get("last_finished_at", "")
            )

    if state.get("status") == "complete":
        found["deepseek-all-configured-wl"] = str(state.get("completed_at", ""))
    return found


def push(label: str) -> bool:
    try:
        completed = subprocess.run(
            [str(PUSH), label],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=1800,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[milestone-watch] {label}: checkpoint command failed: {exc}", flush=True)
        return False
    output = (completed.stdout or "").strip()
    if output:
        print(output[-4000:], flush=True)
    return completed.returncode == 0


def main() -> int:
    marker = load_marker()
    initialized = bool(marker.get("initialized"))
    marker.setdefault("initialized", True)

    while True:
        state = read_json(STATE)
        if state is not None:
            current = milestones(state)
            if not initialized:
                marker["events"] = current
                save_marker(marker)
                initialized = True
                if current:
                    print(
                        "[milestone-watch] baseline established for "
                        + ", ".join(sorted(current)),
                        flush=True,
                    )
            else:
                previous = marker.setdefault("events", {})
                for label, fingerprint in sorted(current.items()):
                    if previous.get(label) == fingerprint:
                        continue
                    print(f"[milestone-watch] detected {label}", flush=True)
                    if push(label):
                        previous[label] = fingerprint
                        save_marker(marker)
                    else:
                        print(
                            f"[milestone-watch] keeping {label} pending for retry",
                            flush=True,
                        )
        time.sleep(INTERVAL)


if __name__ == "__main__":
    raise SystemExit(main())
