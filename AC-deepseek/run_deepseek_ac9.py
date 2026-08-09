#!/usr/bin/env python3
"""Persistent DeepSeek AC runner: one shared warmup, then three parallel WLs."""
from __future__ import annotations

import json
import os
import random
import subprocess
import time
from pathlib import Path

from scheduler.run_pipeline import (
    AC_REPO,
    RESULTS as PIPELINE_RESULTS,
    VENV_PY,
    EscalatingBackoff,
    ac_command,
    ac_env,
    ac_session_complete,
    ensure_ac_shared_warmup,
    load_llm_environment,
    load_state,
    prepare_ac_worldline,
    worldline_panel,
    write_run_config,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "ac9wl_deepseek"
LOGS = RESULTS / "logs"
STATE_PATH = RESULTS / "run_state.json"
CADENCE = 10
WARMUP_MAX_CYCLES = 40
ONLINE_MAX_CYCLES = 300
RETRY_DELAYS = (60, 120, 300, 600, 900)
RETRY_JITTER = 0.20
ONLINE_WORLDLINES = 3
MINER_RETRY_ATTEMPTS = 4
MINER_RETRY_DELAYS = "15,30,60"
MINER_RETRY_429_ATTEMPTS = 10
MINER_RETRY_429_DELAYS = "15,30,45,60,90,120,180,300,600,900"
MINER_RETRY_COMPAT_ATTEMPTS = 8
MINER_RETRY_COMPAT_DELAYS = "5,10,20,40,80,160,300,600"
MINER_RETRY_JITTER = 0.20
REQUEST_TIMEOUT_SECONDS = 180
MILESTONE_PUSH = ROOT.parent / "ops" / "git_milestone_push.sh"


def save_state(state: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)


def push_milestone(label: str) -> None:
    """Create and push a result checkpoint without stopping the experiment."""
    try:
        result = subprocess.run(
            [str(MILESTONE_PUSH), label],
            cwd=ROOT.parent,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=1800,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[deepseek-ac] milestone {label} checkpoint failed: {exc}", flush=True)
        return
    output = (result.stdout or "").strip()
    if output:
        print(output[-4000:], flush=True)
    if result.returncode != 0:
        print(
            f"[deepseek-ac] milestone {label} push failed; periodic saver will retry",
            flush=True,
        )


def state_for_wl(state: dict, wl: int) -> dict:
    return state.setdefault(f"wl{wl}", {})


def start_worldline(wl: int, config: Path, state: dict):
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"wl{wl}.log"
    log = log_path.open("a", encoding="utf-8", buffering=1)
    log.write(f"\n=== launch WL{wl} {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    proc = subprocess.Popen(
        ac_command(f"wl{wl}", config),
        cwd=str(AC_REPO),
        env=ac_env(CADENCE),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    entry = state_for_wl(state, wl)
    entry.update({
        "status": "running",
        "pid": proc.pid,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": str(config),
    })
    save_state(state)
    return proc, log


def main() -> int:
    load_llm_environment()
    if not os.environ.get("OPENAI_API_URL") or not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("deepseek AC: OPENAI_API_URL and OPENAI_API_KEY are required")

    # Keep retries local to the DeepSeek copy.  A failed Miner is retried from
    # its current cycle's initial context; only after this budget is exhausted
    # does main.py exit non-zero, allowing run_pipeline's --resume retry to
    # restart the same cycle without advancing Screener/Trader.
    os.environ.setdefault("AC_DEEPSEEK_RETRY", "1")
    # The local Miner classifier owns 429/502/503/compatibility retries;
    # leave only a small SDK retry budget underneath it to avoid multiplying
    # long waits at both layers.
    os.environ.setdefault("AC_OPENAI_MAX_RETRIES", "2")
    os.environ.setdefault(
        "AC_DEEPSEEK_MINER_RETRY_ATTEMPTS", str(MINER_RETRY_ATTEMPTS)
    )
    os.environ.setdefault("AC_DEEPSEEK_MINER_RETRY_DELAYS", MINER_RETRY_DELAYS)
    os.environ.setdefault(
        "AC_DEEPSEEK_MINER_RETRY_429_ATTEMPTS", str(MINER_RETRY_429_ATTEMPTS)
    )
    os.environ.setdefault(
        "AC_DEEPSEEK_MINER_RETRY_429_DELAYS", MINER_RETRY_429_DELAYS
    )
    os.environ.setdefault(
        "AC_DEEPSEEK_MINER_RETRY_COMPAT_ATTEMPTS", str(MINER_RETRY_COMPAT_ATTEMPTS)
    )
    os.environ.setdefault(
        "AC_DEEPSEEK_MINER_RETRY_COMPAT_DELAYS", MINER_RETRY_COMPAT_DELAYS
    )
    os.environ.setdefault(
        "AC_DEEPSEEK_MINER_RETRY_JITTER", str(MINER_RETRY_JITTER)
    )
    os.environ.setdefault(
        "AC_DEEPSEEK_REQUEST_TIMEOUT", str(REQUEST_TIMEOUT_SECONDS)
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    state = load_state(STATE_PATH)
    state.update({
        "schema_version": 1,
        "model": "deepseek-v4-flash",
        "api_url": os.environ["OPENAI_API_URL"],
        "transport": "native Responses ingress via sub2api; account-level Chat upstream bridge",
        "miner_retry": {
            "enabled": True,
            "generic_attempts": int(os.environ["AC_DEEPSEEK_MINER_RETRY_ATTEMPTS"]),
            "429_attempts": int(os.environ["AC_DEEPSEEK_MINER_RETRY_429_ATTEMPTS"]),
            "compatibility_attempts": int(
                os.environ["AC_DEEPSEEK_MINER_RETRY_COMPAT_ATTEMPTS"]
            ),
            "429_delays_seconds": os.environ["AC_DEEPSEEK_MINER_RETRY_429_DELAYS"],
            "compatibility_delays_seconds": os.environ[
                "AC_DEEPSEEK_MINER_RETRY_COMPAT_DELAYS"
            ],
            "jitter_fraction": float(os.environ["AC_DEEPSEEK_MINER_RETRY_JITTER"]),
            "request_timeout_seconds": float(
                os.environ["AC_DEEPSEEK_REQUEST_TIMEOUT"]
            ),
            "fail_closed_cycle": True,
        },
        "data_root": os.environ.get("AC_DATA_ROOT"),
        "cadence": CADENCE,
        "warmup_max_cycles": WARMUP_MAX_CYCLES,
        "online_max_cycles": ONLINE_MAX_CYCLES,
        "parallel_worldlines": ONLINE_WORLDLINES,
        "run_started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    save_state(state)
    push_milestone("deepseek-run-start")

    print("[deepseek-ac] starting shared 40-cycle warmup", flush=True)
    warmup_ok, manifest = ensure_ac_shared_warmup(
        worldline_panel(1), CADENCE, EscalatingBackoff(), 0
    )
    if not warmup_ok:
        print("[deepseek-ac] shared warmup failed", flush=True)
        return 1
    state["shared_warmup"] = {
        "status": "complete",
        "manifest": str(PIPELINE_RESULTS / "ac" / "shared_warmup" / "manifest.json"),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_state(state)
    push_milestone("deepseek-warmup-complete")

    print(f"[deepseek-ac] seeding WL1..WL{ONLINE_WORLDLINES} from shared warmup", flush=True)
    for wl in range(1, ONLINE_WORLDLINES + 1):
        entry = state_for_wl(state, wl)
        if entry.get("seeded"):
            continue
        seed_state = prepare_ac_worldline(wl, worldline_panel(wl), manifest, CADENCE)
        entry.update({"seeded": True, "seed_state": seed_state})
        save_state(state)

    config = write_run_config(ONLINE_MAX_CYCLES, RESULTS / "run_config.yaml")
    active: dict[int, tuple[subprocess.Popen, object]] = {}
    retry_at: dict[int, float] = {}
    retry_attempt: dict[int, int] = {}
    for wl in range(1, ONLINE_WORLDLINES + 1):
        if ac_session_complete(f"wl{wl}"):
            state_for_wl(state, wl).update({"status": "complete", "ac_done": True})
        else:
            retry_at[wl] = 0.0
    save_state(state)

    print(f"[deepseek-ac] launching {ONLINE_WORLDLINES} WL processes in parallel", flush=True)
    while retry_at or active:
        now = time.monotonic()
        for wl in list(retry_at):
            if wl in active or now < retry_at[wl]:
                continue
            active[wl] = start_worldline(wl, config, state)
            del retry_at[wl]

        for wl, (proc, log) in list(active.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            entry = state_for_wl(state, wl)
            complete = ac_session_complete(f"wl{wl}")
            entry.update({
                "returncode": rc,
                "last_finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ac_done": complete,
            })
            del active[wl]
            if complete:
                entry["status"] = "complete"
                retry_attempt.pop(wl, None)
                print(f"[deepseek-ac] WL{wl} complete", flush=True)
                save_state(state)
                push_milestone(f"deepseek-wl{wl}")
            else:
                entry["status"] = "retry_wait"
                attempt = retry_attempt.get(wl, 0)
                base_delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                delay = base_delay * random.uniform(
                    1.0 - RETRY_JITTER, 1.0 + RETRY_JITTER
                )
                retry_attempt[wl] = attempt + 1
                retry_at[wl] = time.monotonic() + delay
                entry["retry_attempt"] = attempt + 1
                entry["retry_delay_seconds"] = round(delay, 2)
                print(
                    f"[deepseek-ac] WL{wl} rc={rc}, retrying in {delay:g}s",
                    flush=True,
                )
            save_state(state)

        time.sleep(10)

    state["status"] = "complete"
    state["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_state(state)
    push_milestone("deepseek-all-configured-wl")
    print("[deepseek-ac] all configured WLs complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
