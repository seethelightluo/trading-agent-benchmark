#!/usr/bin/env python3
"""Pause the least valuable Terra WLs when the relay is repeatedly rate-limited.

The relay log is global, so this is intentionally a concurrency guard rather
than a claim that one particular WL caused each response.  It starts from the
current log end, counts only newly appended HTTP 429 responses, pauses WL5
first, and pauses WL4 only if the rate remains high after WL5 is stopped.
"""
from __future__ import annotations

import json
import os
import re
import signal
import time
from pathlib import Path


RUN_DIR = Path(os.environ.get(
    "AC_LUNA_RUN_DIR",
    "/home/lxx/trade-agent-benchmark/agent-framework/results/ac_luna_3wl_v4",
)).resolve()
RELAY_LOG = Path(os.environ.get("AC_RELAY_LOG", "/home/lxx/ac-llm-relay/relay.log"))
STATE_PATH = RUN_DIR / "429_monitor.json"
WINDOW_SECONDS = int(os.environ.get("AC_429_WINDOW_SECONDS", "120"))
THRESHOLD = int(os.environ.get("AC_429_THRESHOLD", "5"))
POLL_SECONDS = int(os.environ.get("AC_429_POLL_SECONDS", "20"))
INITIAL_PAUSE_LEVEL = int(os.environ.get("AC_429_INITIAL_PAUSE_LEVEL", "1"))
RE_429 = re.compile(r'"POST /v1/responses HTTP/1\.1" 429\b')


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            pass
    return {
        "schema_version": 1,
        "window_seconds": WINDOW_SECONDS,
        "threshold": THRESHOLD,
        "pause_level": INITIAL_PAUSE_LEVEL,
        "events": [],
        "log_offset": 0,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_state(state: dict) -> None:
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STATE_PATH)


def read_new_lines(offset: int) -> tuple[int, list[str]]:
    try:
        with RELAY_LOG.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            text = handle.read()
            return handle.tell(), text.splitlines()
    except OSError:
        return offset, []


def pid_for(wl: int) -> int | None:
    try:
        state = json.loads((RUN_DIR / "run_state.json").read_text(encoding="utf-8"))
        pid = int(state.get(f"wl{wl}", {}).get("pid") or 0)
        return pid or None
    except (OSError, ValueError, TypeError):
        return None


def pause_wl(wl: int, state: dict, reason: str) -> None:
    marker = RUN_DIR / f"pause_wl{wl}_429"
    marker.touch(exist_ok=True)
    pid = pid_for(wl)
    if pid:
        try:
            os.kill(pid, signal.SIGSTOP)
        except ProcessLookupError:
            pid = None
        except PermissionError:
            pass
    state.setdefault("pauses", []).append({
        "wl": wl,
        "pid": pid,
        "reason": reason,
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    state["pause_level"] = max(int(state.get("pause_level", 0)), 6 - wl)
    print(f"[terra-429] paused WL{wl} pid={pid} reason={reason}", flush=True)


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    try:
        state["log_offset"] = RELAY_LOG.stat().st_size
    except OSError:
        state["log_offset"] = 0
    save_state(state)
    print(
        f"[terra-429] watching {RELAY_LOG} window={WINDOW_SECONDS}s "
        f"threshold={THRESHOLD} initial_pause_level={state.get('pause_level', 0)}",
        flush=True,
    )
    while True:
        offset, lines = read_new_lines(int(state.get("log_offset", 0)))
        state["log_offset"] = offset
        now = time.time()
        events = [float(value) for value in state.get("events", []) if now - float(value) <= WINDOW_SECONDS]
        events.extend(now for line in lines if RE_429.search(line))
        state["events"] = events
        recent = len(events)
        state["last_check_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state["recent_429"] = recent
        level = int(state.get("pause_level", 0))
        if recent >= THRESHOLD and level < 1:
            pause_wl(5, state, f"{recent} relay 429 in {WINDOW_SECONDS}s")
        elif recent >= THRESHOLD and level < 2:
            pause_wl(4, state, f"{recent} relay 429 remained after WL5 pause")
        save_state(state)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
