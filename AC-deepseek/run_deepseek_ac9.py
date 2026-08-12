#!/usr/bin/env python3
"""Persistent DeepSeek AC runner: one shared warmup, then three parallel WLs."""
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
    VENV_PY,
    EscalatingBackoff,
    ac_command,
    ac_env,
    ac_session_complete,
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
MAX_RETRY_ATTEMPTS = int(os.environ.get("AC_DEEPSEEK_MAX_RETRIES", "6"))
ONLINE_WORLDLINES = int(os.environ.get("AC_DEEPSEEK_WORLDLINES", "9"))
if not 1 <= ONLINE_WORLDLINES <= 9:
    raise SystemExit("AC_DEEPSEEK_WORLDLINES must be between 1 and 9")
MAX_CONCURRENT_WL = int(os.environ.get("AC_DEEPSEEK_CONCURRENCY", "3"))
if not 1 <= MAX_CONCURRENT_WL <= ONLINE_WORLDLINES:
    raise SystemExit("AC_DEEPSEEK_CONCURRENCY must be between 1 and ONLINE_WORLDLINES")
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


def pause_marker(wl: int) -> Path:
    """Persistent operator/stall pause marker for one worldline."""
    return RESULTS / f"pause_wl{wl}_429"


def is_paused(wl: int) -> bool:
    return pause_marker(wl).exists()


def push_milestone(label: str) -> None:
    """Create and push a result checkpoint when explicitly enabled.

    A foreground/network-blocking git fetch must never hold the experiment
    supervisor before it can resume a checkpoint.  The operator can enable
    this only after configuring a path-scoped publisher with
    ``AC_DEEPSEEK_ENABLE_MILESTONE_PUSH=1``.
    """
    if os.environ.get("AC_DEEPSEEK_ENABLE_MILESTONE_PUSH") != "1":
        print(
            f"[deepseek-ac] milestone {label} deferred "
            "(non-blocking path-scoped push disabled)",
            flush=True,
        )
        return
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


def load_persisted_warmup() -> tuple[bool, dict]:
    """Validate the existing DeepSeek warm-up without making an API call."""
    from scheduler.ac_shared_warmup import (
        validate_warmup_workspace,
        workflow_cycle_complete,
        workspace_digest,
    )

    manifest_path = PIPELINE_RESULTS / "ac" / "shared_warmup" / "manifest.json"
    manifest = load_state(manifest_path)
    session_dir = AC_REPO / "sandbox" / "ws1"
    if manifest.get("status") != "ready" or manifest.get("session") != "ws1":
        print(f"[deepseek-ac] persisted warmup manifest is not ready: {manifest_path}", flush=True)
        return False, {}
    try:
        artifacts = validate_warmup_workspace(session_dir)
        config = yaml.safe_load((AC_REPO / "config.yaml").read_text(encoding="utf-8"))
        miner_ids = list(config["miner"]["ids"])
        date_state = load_state(session_dir / "persistent" / "date.json")
        account = load_state(session_dir / "persistent" / "account.json")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"[deepseek-ac] persisted warmup validation failed: {exc}", flush=True)
        return False, {}
    checks = {
        "workflow_complete": workflow_cycle_complete(session_dir, miner_ids),
        "workspace_digest": artifacts.get("workspace_digest") == manifest.get("workspace_digest"),
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
    if not all(checks.values()):
        print(f"[deepseek-ac] persisted warmup rejected: {checks}", flush=True)
        return False, {}
    print(
        "[deepseek-ac] reusing persisted shared warmup without remine: "
        f"fingerprint={manifest.get('warmup_fingerprint', '')[:16]} "
        f"workspace={workspace_digest(session_dir / 'workspace')[:16]}",
        flush=True,
    )
    return True, manifest


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
        "max_concurrent_wl": MAX_CONCURRENT_WL,
        "run_started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    save_state(state)
    push_milestone("deepseek-run-start")

    # This copy is online-only.  It must never recreate or extend warmup from
    # a monitor, a stale environment variable, or a manual restart.  The
    # persisted artifact is validated locally; an invalid artifact is a hard
    # stop so recovery cannot silently consume new LLM warmup calls.
    print("[deepseek-ac] validating and reusing persisted shared warmup", flush=True)
    warmup_ok, manifest = load_persisted_warmup()
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
        if is_paused(wl):
            state_for_wl(state, wl).update({"status": "paused_429", "paused_marker": str(pause_marker(wl))})
        elif ac_session_complete(f"wl{wl}"):
            state_for_wl(state, wl).update({"status": "complete", "ac_done": True})
        else:
            retry_at[wl] = 0.0
    save_state(state)

    print(f"[deepseek-ac] launching WL1..WL{ONLINE_WORLDLINES} with max {MAX_CONCURRENT_WL} concurrent", flush=True)
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
            if len(active) >= MAX_CONCURRENT_WL:
                break
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
                if is_paused(wl):
                    entry.update({"status": "paused_429", "paused_marker": str(pause_marker(wl))})
                    retry_message = "paused by operator/stall marker"
                else:
                    attempt = retry_attempt.get(wl, 0)
                    if attempt >= MAX_RETRY_ATTEMPTS:
                        # Retry budget exhausted: auto-pause this WL so queued
                        # WLs (e.g. wl6-9) can take its slot. Operator must
                        # remove the marker to resume it manually.
                        marker = pause_marker(wl)
                        marker.write_text(
                            f"auto-paused at {time.strftime('%Y-%m-%d %H:%M:%S')} "
                            f"after {attempt} retries (budget {MAX_RETRY_ATTEMPTS})\n",
                            encoding="utf-8",
                        )
                        entry.update({
                            "status": "paused_retry_budget",
                            "paused_marker": str(marker),
                            "retry_attempt": attempt,
                        })
                        retry_message = f"auto-paused after {attempt} retries (budget exhausted)"
                        print(
                            f"[deepseek-ac] WL{wl} rc={rc}, {retry_message}",
                            flush=True,
                        )
                        save_state(state)
                        continue
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
                    f"[deepseek-ac] WL{wl} rc={rc}, {retry_message}",
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
