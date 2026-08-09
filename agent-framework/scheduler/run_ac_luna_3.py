#!/usr/bin/env python3
"""Run the original Luna/Terra AC copy with three parallel worldlines.

The shared ws1 warm-up is reused only after the scheduler's fingerprint,
frozen-date, account, workflow, and factor-artifact checks pass. WL1-WL3 are
then seeded independently and kept in separate resumable AC sessions.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import yaml

from scheduler.run_pipeline import (
    AC_REPO,
    RESULTS as PIPELINE_RESULTS,
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


HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
RESULTS = HERE / "results" / "ac_luna_3wl"
LOGS = RESULTS / "logs"
STATE_PATH = RESULTS / "run_state.json"
MILESTONE_PUSH = ROOT / "ops" / "git_milestone_push.sh"
CADENCE = 10
ONLINE_MAX_CYCLES = 300
ONLINE_WORLDLINES = 3
RETRY_SECONDS = 60


def save_state(state: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STATE_PATH)


def push_milestone(label: str) -> None:
    """Checkpoint without turning a transient Git/network error into AC loss."""
    try:
        result = subprocess.run(
            [str(MILESTONE_PUSH), label],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=1800,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[luna-ac] milestone {label} checkpoint failed: {exc}", flush=True)
        return
    output = (result.stdout or "").strip()
    if output:
        print(output[-4000:], flush=True)
    if result.returncode != 0:
        print(
            f"[luna-ac] milestone {label} push failed; periodic saver will retry",
            flush=True,
        )


def state_for_wl(state: dict, wl: int) -> dict:
    return state.setdefault(f"wl{wl}", {})


def start_worldline(wl: int, config: Path, state: dict):
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"wl{wl}.log"
    log = log_path.open("a", encoding="utf-8", buffering=1)
    log.write(f"\n=== launch Luna WL{wl} {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    proc = subprocess.Popen(
        ac_command(f"wl{wl}", config),
        cwd=str(AC_REPO),
        env=ac_env(CADENCE),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    entry = state_for_wl(state, wl)
    entry.update(
        {
            "status": "running",
            "pid": proc.pid,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": str(config),
        }
    )
    save_state(state)
    return proc, log


def main() -> int:
    load_llm_environment()
    if not os.environ.get("OPENAI_API_URL") or not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Luna AC: OPENAI_API_URL and OPENAI_API_KEY are required")

    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    config_source = yaml.safe_load((AC_REPO / "config.yaml").read_text(encoding="utf-8"))
    model = config_source["miner"]["model"]["code"]
    state = load_state(STATE_PATH)
    state.update(
        {
            "schema_version": 1,
            "model": model,
            "api_url": os.environ["OPENAI_API_URL"],
            "transport": "original AC relay configuration",
            "data_root": os.environ.get("AC_DATA_ROOT"),
            "cadence": CADENCE,
            "online_max_cycles": ONLINE_MAX_CYCLES,
            "parallel_worldlines": ONLINE_WORLDLINES,
            "run_started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    save_state(state)
    push_milestone("luna-3wl-started")

    print("[luna-ac] verifying and reusing shared 40-cycle warmup", flush=True)
    warmup_ok, manifest = ensure_ac_shared_warmup(
        worldline_panel(1), CADENCE, EscalatingBackoff(), 0
    )
    if not warmup_ok:
        print("[luna-ac] shared warmup validation failed; no WL launched", flush=True)
        state["status"] = "warmup_failed"
        save_state(state)
        return 1

    state["shared_warmup"] = {
        "status": "verified",
        "manifest": str(PIPELINE_RESULTS / "ac" / "shared_warmup" / "manifest.json"),
        "verified_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warmup_fingerprint": manifest.get("warmup_fingerprint"),
        "workspace_digest": manifest.get("workspace_digest"),
    }
    save_state(state)
    push_milestone("luna-warmup-verified")

    print(f"[luna-ac] seeding WL1..WL{ONLINE_WORLDLINES}", flush=True)
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
    for wl in range(1, ONLINE_WORLDLINES + 1):
        if ac_session_complete(f"wl{wl}"):
            state_for_wl(state, wl).update({"status": "complete", "ac_done": True})
        else:
            retry_at[wl] = 0.0
    save_state(state)

    print(f"[luna-ac] launching {ONLINE_WORLDLINES} WL processes in parallel", flush=True)
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
            entry.update(
                {
                    "returncode": rc,
                    "last_finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "ac_done": complete,
                }
            )
            del active[wl]
            if complete:
                entry["status"] = "complete"
                print(f"[luna-ac] WL{wl} complete", flush=True)
                save_state(state)
                push_milestone(f"luna-wl{wl}-complete")
            else:
                entry["status"] = "retry_wait"
                retry_at[wl] = time.monotonic() + RETRY_SECONDS
                print(
                    f"[luna-ac] WL{wl} rc={rc}, retrying in {RETRY_SECONDS}s",
                    flush=True,
                )
                save_state(state)
        time.sleep(10)

    state["status"] = "complete"
    state["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_state(state)
    push_milestone("luna-all-configured-wl-complete")
    print("[luna-ac] all configured WLs complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
