#!/usr/bin/env python3
"""Hourly, fail-closed AC Terra/DeepSeek operations monitor.

This monitor is deliberately online-only.  It observes the two AC copies,
repairs the local transport when the transport is actually broken, and only
restarts a supervisor from an existing online checkpoint.  It never calls a
warmup helper and never deletes persistent AC state.  Recovery snapshots are
stored outside the repository under ``/data/ac-hourly-snapshots``.

The systemd timer invokes this file once per hour.  Three consecutive healthy
observations mark the monitor complete and disable its timer.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/home/lxx/trade-agent-benchmark")
VENV_PY = ROOT / ".venv" / "bin" / "python"
FRAMEWORK = ROOT / "agent-framework"
AC_REPO = FRAMEWORK / "AlphaCrafter" / "alphacrafter"
TERRA_RESULTS = FRAMEWORK / "results" / "ac_luna_3wl_v4"
TERRA_SANDBOX = AC_REPO / "sandbox"
DS_ROOT = ROOT / "AC-deepseek"
DS_RESULTS = DS_ROOT / "results" / "ac9wl_deepseek"
DS_SANDBOX = DS_ROOT / "AlphaCrafter" / "alphacrafter" / "sandbox"
DATA_ROOT = ROOT / "WL-data-final"
CLASH_CONFIG = Path(
    "/home/lxx/.local/share/io.github.clash-verge-rev.clash-verge-rev/clash-verge.yaml"
)
CLASH_SOCKET = Path("/tmp/verge/verge-mihomo.sock")
RELAY_LOG = Path("/home/lxx/ac-llm-relay/relay.log")
MONITOR_ROOT = Path("/data/ac-hourly-monitor")
STATE_PATH = MONITOR_ROOT / "state.json"
LOG_PATH = MONITOR_ROOT / "monitor.log"
LOCK_PATH = MONITOR_ROOT / "monitor.lock"
SNAPSHOT_ROOT = Path("/data/ac-hourly-snapshots")
TIMER_UNIT = "ac-hourly-monitor.timer"

TERRA_WLS = (1, 2, 3)
TERRA_PREFIX = "terra_v4_wl"
DS_WLS = (1, 2, 3)
STALL_SECONDS = 3900
ERROR_WINDOW_BYTES = 4 * 1024 * 1024
MAX_SNAPSHOTS = 5

TRANSIENT_RE = re.compile(
    r"(?:error code|status_code|http/1\.1)\D*(?:429|502|503)\b|"
    r"pool exhausted|service temporarily unavailable|timeout|connection refused|"
    r"connection reset|broken pipe|unexpected eof|internal_error|"
    r"temporarily unavailable|proxyconnect",
    re.IGNORECASE,
)
FATAL_RE = re.compile(
    r"traceback \(most recent call last\)|filenotfounderror|permissionerror|"
    r"zerodivisionerror|workflow failed|warmup failed|persisted warmup rejected|"
    r"no such file",
    re.IGNORECASE,
)
TRANSPORT_RE = re.compile(
    r"(?:error code|status_code|http/1\.1)\D*(?:502|503)\b|proxyconnect|"
    r"connection refused|connection reset|broken pipe|unexpected eof|"
    r"timeout|internal_error|temporarily unavailable",
    re.IGNORECASE,
)


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    MONITOR_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"[{now_text()}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def command_line(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except (OSError, ValueError):
        return ""


def alive(pid: object) -> bool:
    try:
        number = int(pid)
    except (TypeError, ValueError):
        return False
    return number > 1 and Path(f"/proc/{number}").exists()


def matching_pids(*needles: str) -> list[int]:
    found: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        cmd = command_line(int(entry.name))
        if cmd and all(needle in cmd for needle in needles):
            found.append(int(entry.name))
    return sorted(found)


def read_cursor(root: Path, session: str) -> dict:
    date = load_json(root / session / "persistent" / "date.json")
    account = load_json(root / session / "persistent" / "account.json")
    return {
        "date": date.get("current_date"),
        "visible_through": date.get("visible_through"),
        "complete": bool(date.get("simulation_complete")),
        "orders": len(account.get("orders", []) or []),
        "positions": len(account.get("positions", {}) or {}),
    }


def file_signature(paths: list[Path]) -> dict[str, int]:
    result: dict[str, int] = {}
    for path in paths:
        try:
            result[str(path)] = path.stat().st_size
        except OSError:
            result[str(path)] = -1
    return result


def process_env(pid: int, key: str) -> str:
    try:
        for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
            if item.startswith(key.encode("utf-8") + b"="):
                return item.split(b"=", 1)[1].decode("utf-8", "replace")
    except OSError:
        pass
    return ""


def run_supervisor_pids(needle: str, *, env_key: str = "", env_value: str = "", cwd: Path | None = None) -> list[int]:
    """Find only this run, excluding stale supervisors from older experiments."""
    result: list[int] = []
    for pid in matching_pids(needle):
        if env_key and process_env(pid, env_key) != env_value:
            continue
        if cwd is not None:
            try:
                if Path(os.readlink(f"/proc/{pid}/cwd")).resolve() != cwd.resolve():
                    continue
            except OSError:
                continue
        result.append(pid)
    return result


def collect_new_errors(state: dict, paths: list[Path]) -> dict:
    offsets = state.setdefault("log_offsets", {})
    if not offsets:
        # Establish a baseline at installation time.  Historical traceback
        # text in old AC logs must not trigger an automatic recovery on the
        # first hourly tick.
        for path in paths:
            try:
                offsets[str(path)] = path.stat().st_size
            except OSError:
                offsets[str(path)] = 0
        return {"transient": 0, "fatal": 0, "samples": [], "baseline": True}
    transient = 0
    fatal = 0
    transport = 0
    samples: list[str] = []
    for path in paths:
        try:
            size = path.stat().st_size
            offset = int(offsets.get(str(path), 0))
            if offset < 0 or offset > size:
                offset = 0
            if size - offset > ERROR_WINDOW_BYTES:
                offset = size - ERROR_WINDOW_BYTES
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                lines = handle.readlines()
            offsets[str(path)] = size
        except (OSError, ValueError):
            continue
        for line in lines:
            if TRANSIENT_RE.search(line):
                transient += 1
            if TRANSPORT_RE.search(line):
                transport += 1
            if FATAL_RE.search(line):
                fatal += 1
            if (TRANSIENT_RE.search(line) or FATAL_RE.search(line)) and len(samples) < 8:
                samples.append(f"{path.name}: {line.strip()[-300:]}")
    return {"transient": transient, "transport": transport, "fatal": fatal, "samples": samples}


def terra_snapshot(state: dict) -> dict:
    run = load_json(TERRA_RESULTS / "run_state.json")
    pids = {
        f"wl{wl}": run.get(f"wl{wl}", {}).get("pid") for wl in TERRA_WLS
    }
    cursors = {
        f"wl{wl}": read_cursor(TERRA_SANDBOX, f"{TERRA_PREFIX}{wl}")
        for wl in TERRA_WLS
    }
    supervisor = run_supervisor_pids(
        "scheduler.run_ac_luna_3",
        env_key="AC_LUNA_RUN_DIR",
        env_value=str(TERRA_RESULTS),
    )
    active = [wl for wl, pid in pids.items() if alive(pid)]
    complete = [wl for wl, cursor in cursors.items() if cursor["complete"]]
    logs = [TERRA_RESULTS / "supervisor.log", RELAY_LOG]
    logs += [TERRA_RESULTS / "logs" / f"wl{wl}.log" for wl in range(1, 6)]
    return {
        "supervisor_pids": supervisor,
        "worker_pids": pids,
        "active_wls": active,
        "complete_wls": complete,
        "cursors": cursors,
        "paused_wls": [wl for wl in (4, 5) if (TERRA_RESULTS / f"pause_wl{wl}_429").exists()],
        "logs": [str(path) for path in logs if path.exists()],
        "process_ok": bool(supervisor),
        "target_ok": bool(active or len(complete) == len(TERRA_WLS)),
    }


def deepseek_snapshot(state: dict) -> dict:
    run = load_json(DS_RESULTS / "run_state.json")
    pids = {f"wl{wl}": run.get(f"wl{wl}", {}).get("pid") for wl in DS_WLS}
    cursors = {f"wl{wl}": read_cursor(DS_SANDBOX, f"wl{wl}") for wl in DS_WLS}
    supervisor = run_supervisor_pids("run_deepseek_ac9", cwd=DS_ROOT)
    active = [wl for wl, pid in pids.items() if alive(pid)]
    complete = [wl for wl, cursor in cursors.items() if cursor["complete"]]
    logs = [DS_RESULTS / "supervisor.log"]
    logs += [DS_RESULTS / "logs" / f"wl{wl}.log" for wl in DS_WLS]
    seeded = any(bool(run.get(f"wl{wl}", {}).get("seeded")) for wl in DS_WLS)
    safe_online_state = bool(run.get("shared_warmup", {}).get("status")) and seeded
    return {
        "supervisor_pids": supervisor,
        "worker_pids": pids,
        "active_wls": active,
        "complete_wls": complete,
        "cursors": cursors,
        "logs": [str(path) for path in logs if path.exists()],
        "process_ok": bool(supervisor),
        "target_ok": bool(active or len(complete) == len(DS_WLS)),
        "safe_online_state": safe_online_state,
        "blocked_reason": "no seeded online checkpoint; warmup is never auto-created" if not safe_online_state else "",
    }


def proxy_listener_ok() -> bool:
    try:
        with socket.create_connection(("172.18.0.1", 7898), timeout=2):
            return True
    except OSError:
        return False


def direct_opencode_ok() -> bool:
    try:
        result = subprocess.run(
            ["curl", "--noproxy", "*", "--connect-timeout", "5", "--max-time", "10", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "https://opencode.ai"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        return False


def ensure_listener_config() -> bool:
    if not CLASH_CONFIG.exists():
        return False
    text = CLASH_CONFIG.read_text(encoding="utf-8")
    if "name: sub2api-codex-jp" in text and "port: 7898" in text:
        return True
    anchor = "bind-address: '*'\n"
    block = (
        "listeners:\n"
        "- name: sub2api-codex-jp\n"
        "  type: mixed\n"
        "  port: 7898\n"
        "  listen: 172.18.0.1\n"
        "  udp: true\n"
        "  proxy: OpenAI\n"
    )
    if anchor not in text:
        return False
    config_backup = MONITOR_ROOT / "proxy-config-backups" / f"clash-{time.strftime('%Y%m%d-%H%M%S')}.yaml"
    config_backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CLASH_CONFIG, config_backup)
    temporary = CLASH_CONFIG.with_suffix(".yaml.tmp")
    temporary.write_text(text.replace(anchor, anchor + block, 1), encoding="utf-8")
    temporary.replace(CLASH_CONFIG)
    return True


def repair_network(state: dict, dry_run: bool, force_reload: bool = False) -> bool:
    listener_before = proxy_listener_ok()
    direct_before = direct_opencode_ok()
    if listener_before and direct_before and not force_reload:
        return True
    log(
        f"transport anomaly: jp_listener={listener_before} "
        f"direct_opencode={direct_before} force_reload={force_reload}"
    )
    if dry_run:
        return False
    if not ensure_listener_config():
        log("proxy repair blocked: listener configuration could not be verified")
        return False
    if CLASH_SOCKET.exists():
        try:
            subprocess.run(
                ["curl", "--silent", "--show-error", "--unix-socket", str(CLASH_SOCKET),
                 "-X", "PUT", "http://localhost/configs?force=true",
                 "-H", "Content-Type: application/json",
                 "--data", json.dumps({"path": str(CLASH_CONFIG)})],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
                check=False,
            )
            time.sleep(2)
        except (OSError, subprocess.TimeoutExpired) as exc:
            log(f"mihomo hot reload failed: {exc}")
    if not proxy_listener_ok():
        try:
            subprocess.run(["systemctl", "restart", "clash-verge-service"], timeout=45, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            log(f"clash service restart failed: {exc}")
        time.sleep(3)
    result = proxy_listener_ok() and direct_opencode_ok()
    log(f"transport repair result: jp_listener={proxy_listener_ok()} direct_opencode={direct_opencode_ok()}")
    return result


def rsync_copy(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["rsync", "-a", "--", str(source) + "/", str(destination) + "/"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1800,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stdout[-1000:])
    except (OSError, subprocess.TimeoutExpired, RuntimeError):
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)


def backup_results(reason: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    destination = SNAPSHOT_ROOT / f"snapshot-{stamp}"
    destination.mkdir(parents=True, exist_ok=False)
    metadata = {"created_at": now_text(), "reason": reason, "source_root": str(ROOT)}
    save_json(destination / "metadata.json", metadata)
    rsync_copy(TERRA_RESULTS, destination / "terra" / "results")
    rsync_copy(DS_RESULTS, destination / "deepseek" / "results")
    for wl in range(1, 6):
        rsync_copy(TERRA_SANDBOX / f"{TERRA_PREFIX}{wl}", destination / "terra" / "sandbox" / f"wl{wl}")
    for wl in range(1, 4):
        rsync_copy(DS_SANDBOX / f"wl{wl}", destination / "deepseek" / "sandbox" / f"wl{wl}")
    rsync_copy(FRAMEWORK / "results" / "ac" / "shared_warmup", destination / "terra" / "shared_warmup")
    rsync_copy(DS_RESULTS.parent / "ac" / "shared_warmup", destination / "deepseek" / "shared_warmup")
    snapshots = sorted(SNAPSHOT_ROOT.glob("snapshot-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in snapshots[MAX_SNAPSHOTS:]:
        if old.is_dir() and old.name.startswith("snapshot-"):
            shutil.rmtree(old)
    log(f"backup created: {destination}")
    return destination


def cleanup_transients(paths: list[Path]) -> int:
    removed = 0
    for root in paths:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".tmp", ".part"}:
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    pass
    return removed


def stop_known_processes(snapshot: dict, kind: str) -> None:
    pids: set[int] = set(snapshot.get("supervisor_pids", []))
    pids.update(int(pid) for pid in snapshot.get("worker_pids", {}).values() if alive(pid))
    needle = "run_ac_luna_3" if kind == "terra" else "run_deepseek_ac9"
    for pid in sorted(pids):
        cmd = command_line(pid)
        if pid not in snapshot.get("supervisor_pids", []) and needle not in cmd and "main.py" not in cmd:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    deadline = time.time() + 20
    while time.time() < deadline and any(alive(pid) for pid in pids):
        time.sleep(1)
    for pid in sorted(pids):
        if alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass


def start_supervisor(kind: str, dry_run: bool) -> bool:
    if dry_run:
        log(f"dry-run: would restart {kind} supervisor from existing online state")
        return False
    if kind == "terra":
        command = [str(VENV_PY), "-u", "-m", "scheduler.run_ac_luna_3"]
        cwd = FRAMEWORK
        env = os.environ.copy()
        env.update({
            "AC_LUNA_RUN_DIR": str(TERRA_RESULTS),
            "AC_LUNA_WORLDLINES": "5",
            "AC_LUNA_SESSION_PREFIX": TERRA_PREFIX,
            "AC_DATA_ROOT": str(DATA_ROOT),
        })
        log_path = TERRA_RESULTS / "supervisor.log"
    else:
        command = [str(DS_ROOT / "run_deepseek_ac9.sh")]
        cwd = DS_ROOT
        env = os.environ.copy()
        env["AC_DEEPSEEK_REUSE_WARMUP"] = "1"
        env["AC_DATA_ROOT"] = str(DATA_ROOT)
        log_path = DS_RESULTS / "supervisor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8", buffering=1)
    handle.write(f"\n=== hourly monitor restart {kind} {now_text()} ===\n")
    subprocess.Popen(command, cwd=str(cwd), env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    log(f"restarted {kind} supervisor")
    return True


def safe_to_restart(kind: str, snapshot: dict, previous: dict) -> bool:
    if kind == "deepseek" and not snapshot.get("safe_online_state"):
        return False
    previous_changed = float(previous.get("last_changed_epoch", time.time()))
    stalled = time.time() - previous_changed >= STALL_SECONDS
    return stalled and not snapshot.get("process_ok")


def experiment_health(kind: str, snapshot: dict, errors: dict, previous: dict) -> tuple[bool, str]:
    if kind == "deepseek" and not snapshot.get("safe_online_state"):
        return False, str(snapshot.get("blocked_reason"))
    if errors["fatal"] and (not snapshot["process_ok"] or not snapshot["target_ok"]):
        return False, f"new fatal-looking log lines={errors['fatal']}"
    if not snapshot["process_ok"] and len(snapshot["complete_wls"]) < (3 if kind == "deepseek" else 3):
        return False, "supervisor missing before a completed online run"
    if not snapshot["target_ok"]:
        return False, "no target WL is running or complete"
    if time.time() - float(previous.get("last_changed_epoch", time.time())) >= STALL_SECONDS:
        return False, f"no process/cursor/log progress for {STALL_SECONDS}s"
    return True, "running or complete with no new fatal errors"


def run_once(dry_run: bool = False) -> int:
    MONITOR_ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("another monitor invocation is active; exiting")
            return 0
        state = load_json(STATE_PATH)
        if state.get("done") and not dry_run:
            log("three healthy rounds already completed; monitor is done")
            return 0
        previous_experiments = state.setdefault("experiments", {})
        terra = terra_snapshot(state)
        deepseek = deepseek_snapshot(state)
        all_logs = [Path(p) for p in terra["logs"] + deepseek["logs"]]
        errors = collect_new_errors(state, all_logs)
        transport_ok = repair_network(
            state,
            dry_run,
            force_reload=bool(errors.get("transport")),
        )
        current = {"terra": terra, "deepseek": deepseek}
        healthy: dict[str, bool] = {}
        reasons: dict[str, str] = {}
        actions: list[str] = []
        for kind, snapshot in current.items():
            previous = previous_experiments.setdefault(kind, {})
            signature = json.dumps({
                "active": snapshot["active_wls"],
                "complete": snapshot["complete_wls"],
                "cursors": snapshot["cursors"],
                "process": snapshot["process_ok"],
                "log_sizes": file_signature([Path(p) for p in snapshot["logs"]]),
            }, sort_keys=True)
            if signature != previous.get("signature"):
                previous["last_changed_epoch"] = time.time()
            previous["signature"] = signature
            healthy[kind], reasons[kind] = experiment_health(kind, snapshot, errors, previous)
            if not transport_ok and kind == "terra":
                healthy[kind] = False
                reasons[kind] = "transport repair did not restore both paths"
            if safe_to_restart(kind, snapshot, previous) and transport_ok:
                backup = backup_results(f"{kind} stalled or supervisor missing; errors={errors}")
                stop_known_processes(snapshot, kind)
                removed = cleanup_transients(
                    [TERRA_RESULTS, TERRA_SANDBOX] if kind == "terra" else [DS_RESULTS, DS_SANDBOX]
                )
                actions.append(f"{kind}: backup={backup} transient_files_removed={removed}")
                start_supervisor(kind, dry_run)
                previous["last_changed_epoch"] = time.time()
        all_healthy = all(healthy.values()) and transport_ok
        streak = int(state.get("healthy_streak", 0))
        state["healthy_streak"] = streak + 1 if all_healthy else 0
        state["last_run_at"] = now_text()
        state["last_status"] = {
            "terra": {"healthy": healthy["terra"], "reason": reasons["terra"], "snapshot": terra},
            "deepseek": {"healthy": healthy["deepseek"], "reason": reasons["deepseek"], "snapshot": deepseek},
            "errors": errors,
            "transport_ok": transport_ok,
            "actions": actions,
        }
        if state["healthy_streak"] >= 3 and not dry_run:
            state["done"] = True
            state["ended_at"] = now_text()
            log("three consecutive healthy rounds completed; disabling hourly timer")
            subprocess.run(["systemctl", "--user", "disable", "--now", TIMER_UNIT], check=False)
        save_json(STATE_PATH, state)
        log(
            f"terra={'healthy' if healthy['terra'] else 'attention'} "
            f"ds={'healthy' if healthy['deepseek'] else 'attention'} "
            f"transport={'ok' if transport_ok else 'failed'} "
            f"new_transient={errors['transient']} new_fatal={errors['fatal']} "
            f"new_transport={errors.get('transport', 0)} "
            f"healthy_streak={state['healthy_streak']} actions={actions or 'none'}"
        )
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="run one hourly check")
    parser.add_argument("--dry-run", action="store_true", help="observe and print; do not repair/restart/save completion")
    args = parser.parse_args()
    return run_once(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
