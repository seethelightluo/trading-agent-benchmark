#!/usr/bin/env python
"""Structured progress/error monitor for the shared FactorMiner warm-up."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent.parent
DEFAULT_STATE = HERE / "results" / "full_warmup_fm_state.json"
DEFAULT_LOG = HERE / "results" / "full_warmup_fm.log"
DEFAULT_PID = HERE / "results" / "full_warmup_fm.pid"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
PROBLEM_RE = re.compile(
    r"(?:\bERROR\b|\bCRITICAL\b|Traceback|Mining error|Aborted!|"
    r"(?:❌|⛔|⚠️)|rc=\d+|timed?\s*out|rate.?limit|quota)",
    re.IGNORECASE,
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _fm_progress(state: dict[str, Any]) -> dict[str, Any]:
    shared = state.get("shared_warmup")
    if not isinstance(shared, dict):
        return {}
    progress = shared.get("fm_progress")
    return progress if isinstance(progress, dict) else {}


def _pid_status(pid_file: Path) -> tuple[int | None, bool]:
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return None, False
    return pid, True


def _read_batches(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    batches: list[dict[str, Any]] = []
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            batches.append(item)
    return batches


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "?"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def recent_problem_lines(log_path: Path, limit: int = 8) -> list[str]:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    found = []
    for raw in lines:
        line = ANSI_RE.sub("", raw).strip()
        if line and PROBLEM_RE.search(line):
            found.append(line)
    return found[-limit:]


def build_snapshot(state_path: Path, log_path: Path, pid_file: Path) -> dict[str, Any]:
    state = _load_json(state_path)
    progress = _fm_progress(state)
    stage_dir = Path(progress["stage_dir"]) if progress.get("stage_dir") else None
    checkpoint_dir = (
        Path(progress["checkpoint_path"])
        if progress.get("checkpoint_path")
        else (stage_dir / "checkpoint" if stage_dir else None)
    )
    loop_state = _load_json(checkpoint_dir / "loop_state.json") if checkpoint_dir else {}
    batches = _read_batches(stage_dir / "mining_batches.jsonl") if stage_dir else []
    last_batch = batches[-1] if batches else {}

    iteration = int(loop_state.get("iteration", last_batch.get("iteration", 0)) or 0)
    target_iterations = int(progress.get("target_iterations", 200) or 200)
    elapsed = sum(float(item.get("elapsed_seconds", 0.0) or 0.0) for item in batches)
    recent = batches[-5:]
    avg_recent = (
        sum(float(item.get("elapsed_seconds", 0.0) or 0.0) for item in recent) / len(recent)
        if recent
        else None
    )
    eta = avg_recent * max(0, target_iterations - iteration) if avg_recent else None
    pid, running = _pid_status(pid_file)
    candidates = sum(int(item.get("candidates", 0) or 0) for item in batches)
    admitted = sum(int(item.get("admitted", 0) or 0) for item in batches)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pid": pid,
        "running": running,
        "phase": progress.get("warmup_stage", "not-started"),
        "iteration": iteration,
        "target_iterations": target_iterations,
        "library_size": int(loop_state.get("library_size", last_batch.get("library_size", 0)) or 0),
        "library_capacity": int(progress.get("library_capacity", 30) or 30),
        "candidates": candidates,
        "admitted": admitted,
        "last_admitted": int(last_batch.get("admitted", 0) or 0),
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        "last_iteration_seconds": (
            float(last_batch.get("elapsed_seconds", 0.0)) if last_batch else None
        ),
        "retry_attempt": _last_retry_attempt(log_path),
        "problems": recent_problem_lines(log_path),
        "stage_dir": str(stage_dir) if stage_dir else None,
    }


def _last_retry_attempt(log_path: Path) -> str | None:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for raw in reversed(lines):
        line = ANSI_RE.sub("", raw).strip()
        if " FM " in line and " attempt " in line:
            return line
    return None


def format_snapshot(snapshot: dict[str, Any]) -> str:
    status = "RUNNING" if snapshot["running"] else "STOPPED/UNKNOWN"
    pid = snapshot["pid"] if snapshot["pid"] is not None else "?"
    iteration = snapshot["iteration"]
    target = snapshot["target_iterations"]
    pct = (100.0 * iteration / target) if target else 0.0
    lines = [
        f"[{snapshot['timestamp']}] FM warmup {status} pid={pid} phase={snapshot['phase']}",
        (
            f"progress: iter {iteration}/{target} ({pct:.1f}%) | "
            f"library {snapshot['library_size']} (完成后 cap={snapshot['library_capacity']}，<=cap 全留) | "
            f"candidates {snapshot['candidates']} | admitted {snapshot['admitted']}"
        ),
        (
            f"timing: elapsed {_format_duration(snapshot['elapsed_seconds'])} | "
            f"last {_format_duration(snapshot['last_iteration_seconds'])} | "
            f"ETA {_format_duration(snapshot['eta_seconds'])}"
        ),
    ]
    if snapshot.get("retry_attempt"):
        lines.append(f"latest attempt: {snapshot['retry_attempt']}")
    if snapshot.get("stage_dir"):
        lines.append(f"artifacts: {snapshot['stage_dir']}")
    if snapshot["problems"]:
        lines.append("recent errors/warnings:")
        lines.extend(f"  {line}" for line in snapshot["problems"])
    else:
        lines.append("recent errors/warnings: none")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--pid-file", type=Path, default=DEFAULT_PID)
    parser.add_argument("--watch", action="store_true", help="持续监控；迭代变化或出现新报错时打印")
    parser.add_argument("--interval", type=float, default=30.0, help="watch 轮询秒数，默认30")
    args = parser.parse_args()

    previous_key = None
    while True:
        snapshot = build_snapshot(args.state, args.log, args.pid_file)
        key = (
            snapshot["running"],
            snapshot["phase"],
            snapshot["iteration"],
            snapshot["library_size"],
            snapshot["retry_attempt"],
            tuple(snapshot["problems"]),
        )
        if not args.watch or key != previous_key:
            if previous_key is not None:
                print()
            print(format_snapshot(snapshot), flush=True)
            previous_key = key
        if not args.watch:
            return 0
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
