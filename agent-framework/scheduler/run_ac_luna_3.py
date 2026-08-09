#!/usr/bin/env python3
"""Run the original Luna/Terra AC copy with configurable parallel worldlines.

The shared ws1 warm-up is reused only after its persisted manifest,
frozen-date, account, workflow, and factor-artifact checks pass. WL1-WL3 are
then seeded independently and kept in separate resumable AC sessions.  The
default remains three; ``AC_LUNA_WORLDLINES`` can extend an existing run
without changing its session prefix or warm-up.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import time
from pathlib import Path

import yaml

from scheduler.run_pipeline import (
    AC_REPO,
    RESULTS as PIPELINE_RESULTS,
    ac_command,
    ac_env,
    ac_session_complete,
    load_llm_environment,
    load_state,
    prepare_ac_worldline,
    worldline_panel,
    write_run_config,
)
from scheduler.ac_shared_warmup import (
    validate_warmup_workspace,
    workflow_cycle_complete,
    workspace_digest,
)


HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
RESULTS = HERE / "results" / "ac_luna_3wl"
if os.environ.get("AC_LUNA_RUN_DIR"):
    RESULTS = Path(os.environ["AC_LUNA_RUN_DIR"]).resolve()
LOGS = RESULTS / "logs"
STATE_PATH = RESULTS / "run_state.json"
MILESTONE_PUSH = ROOT / "ops" / "git_milestone_push.sh"
CADENCE = 10
ONLINE_MAX_CYCLES = 300
ONLINE_WORLDLINES = int(os.environ.get("AC_LUNA_WORLDLINES", "3"))
if not 1 <= ONLINE_WORLDLINES <= 9:
    raise SystemExit("AC_LUNA_WORLDLINES must be between 1 and 9")
RETRY_DELAYS = (60, 120, 300, 600, 900)
RETRY_JITTER = 0.20
SESSION_PREFIX = os.environ.get("AC_LUNA_SESSION_PREFIX", "wl")


def session_name(wl: int) -> str:
    return f"{SESSION_PREFIX}{wl}"


def save_state(state: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STATE_PATH)


def push_milestone(label: str) -> None:
    """Checkpoint only when explicitly enabled for this isolated run.

    The repository currently contains unrelated warmup/DeepSeek/runtime
    changes.  The legacy helper uses ``git add -A``; invoking it from this
    scheduler would therefore commit other experiments.  Keep milestone
    pushes opt-in until a path-scoped publisher is used.
    """
    if os.environ.get("AC_LUNA_ENABLE_MILESTONE_PUSH") != "1":
        print(f"[luna-ac] milestone {label} deferred (path-scoped push disabled)", flush=True)
        return
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


def pause_marker(wl: int) -> Path:
    """Persistent operator/429 pause marker for one worldline."""
    return RESULTS / f"pause_wl{wl}_429"


def is_paused(wl: int) -> bool:
    return pause_marker(wl).exists()


def verify_persisted_warmup() -> tuple[bool, dict]:
    """Validate the completed Luna artifact without invoking the warmup runner.

    The shared warmup was produced under a historical AC/code fingerprint. A
    later code fingerprint mismatch must not silently archive and remine this
    accepted artifact; only the artifact's own manifest and workspace digest
    decide whether online seeding is allowed.
    """
    manifest_path = PIPELINE_RESULTS / "ac" / "shared_warmup" / "manifest.json"
    manifest = load_state(manifest_path)
    session_dir = AC_REPO / "sandbox" / "ws1"
    if not manifest or manifest.get("status") != "ready":
        print("[luna-ac] persisted warmup manifest is not ready", flush=True)
        return False, {}
    if manifest.get("session") != "ws1":
        print("[luna-ac] persisted warmup session is not ws1", flush=True)
        return False, {}

    try:
        artifacts = validate_warmup_workspace(session_dir)
    except (OSError, ValueError) as exc:
        print(f"[luna-ac] persisted warmup artifacts invalid: {exc}", flush=True)
        return False, {}

    config = yaml.safe_load((AC_REPO / "config.yaml").read_text(encoding="utf-8"))
    miner_ids = list(config["miner"]["ids"])
    date_state = load_state(session_dir / "persistent" / "date.json")
    account = load_state(session_dir / "persistent" / "account.json")
    checks = {
        "workflow_complete": workflow_cycle_complete(session_dir, miner_ids),
        "workspace_artifacts": bool(artifacts.get("factor_files"))
        and artifacts.get("factor_count", 0) <= 30,
        "manifest_factor_files": artifacts.get("factor_files") == manifest.get("factor_files"),
        "baseline_date": date_state.get("current_date") == manifest.get("baseline_date"),
        "history_end": date_state.get("visible_through") == manifest.get("history_end"),
        "simulation_frozen": not date_state.get("simulation_complete"),
        "no_positions": not account.get("positions"),
        "no_orders": not account.get("orders"),
        "initial_capital": float(account.get("initial_capital", 0.0))
        == float(manifest.get("initial_capital_usd", 0.0)),
        "available_cash": float(account.get("available_cash", 0.0))
        == float(manifest.get("initial_capital_usd", 0.0)),
    }
    ensemble = load_state(session_dir / "workspace" / "factors" / "factor_ensemble.json")
    selected = ensemble.get("selected_factors", [])
    checks["active_ensemble"] = (
        0 < len(selected) <= 10
        and abs(sum(float(item.get("weight", 0.0)) for item in selected) - 1.0) < 1e-6
    )
    if not all(checks.values()):
        print(f"[luna-ac] persisted warmup check failed: {checks}", flush=True)
        return False, {}
    print(
        "[luna-ac] persisted warmup accepted without remine: "
        f"fingerprint={manifest.get('warmup_fingerprint', '')[:16]} "
        f"workspace={workspace_digest(session_dir / 'workspace')[:16]} "
        "(manifest metadata may predate the final fractional-sizing patch)",
        flush=True,
    )
    return True, manifest


def start_worldline(wl: int, config: Path, state: dict):
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"wl{wl}.log"
    log = log_path.open("a", encoding="utf-8", buffering=1)
    log.write(
        f"\n=== launch Luna WL{wl} session={session_name(wl)} "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
    )
    proc = subprocess.Popen(
        ac_command(session_name(wl), config),
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
    warmup_ok, manifest = verify_persisted_warmup()
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
        "manifest_workspace_digest": manifest.get("workspace_digest"),
        "workspace_digest": workspace_digest(AC_REPO / "sandbox" / "ws1" / "workspace"),
    }
    save_state(state)
    push_milestone("luna-warmup-verified")

    print(f"[luna-ac] seeding WL1..WL{ONLINE_WORLDLINES}", flush=True)
    for wl in range(1, ONLINE_WORLDLINES + 1):
        entry = state_for_wl(state, wl)
        if entry.get("seeded"):
            continue
        seed_state = prepare_ac_worldline(
            wl,
            worldline_panel(wl),
            manifest,
            CADENCE,
            session_name=session_name(wl),
        )
        entry.update({"seeded": True, "seed_state": seed_state})
        save_state(state)

    config = write_run_config(ONLINE_MAX_CYCLES, RESULTS / "run_config.yaml")
    active: dict[int, tuple[subprocess.Popen, object]] = {}
    retry_at: dict[int, float] = {}
    retry_attempt: dict[int, int] = {}
    for wl in range(1, ONLINE_WORLDLINES + 1):
        if is_paused(wl):
            state_for_wl(state, wl).update({"status": "paused_429", "paused_marker": str(pause_marker(wl))})
        elif ac_session_complete(session_name(wl)):
            state_for_wl(state, wl).update({"status": "complete", "ac_done": True})
        else:
            retry_at[wl] = 0.0
    save_state(state)

    print(f"[luna-ac] launching {ONLINE_WORLDLINES} WL processes in parallel", flush=True)
    while retry_at or active:
        now = time.monotonic()
        for wl in list(retry_at):
            if is_paused(wl):
                state_for_wl(state, wl).update({"status": "paused_429", "paused_marker": str(pause_marker(wl))})
                del retry_at[wl]
                save_state(state)
                continue
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
            complete = ac_session_complete(session_name(wl))
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
                retry_attempt.pop(wl, None)
                print(f"[luna-ac] WL{wl} complete", flush=True)
                save_state(state)
                push_milestone(f"luna-wl{wl}-complete")
            else:
                entry["status"] = "retry_wait"
                if is_paused(wl):
                    entry.update({"status": "paused_429", "paused_marker": str(pause_marker(wl))})
                    retry_message = "paused by 429 marker"
                else:
                    attempt = retry_attempt.get(wl, 0)
                    base_delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                    delay = base_delay * random.uniform(
                        1.0 - RETRY_JITTER, 1.0 + RETRY_JITTER
                    )
                    retry_attempt[wl] = attempt + 1
                    retry_at[wl] = time.monotonic() + delay
                    entry["retry_attempt"] = attempt + 1
                    entry["retry_delay_seconds"] = round(delay, 2)
                    retry_message = f"retrying in {delay:g}s"
                print(
                    f"[luna-ac] WL{wl} rc={rc}, "
                    + retry_message,
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
