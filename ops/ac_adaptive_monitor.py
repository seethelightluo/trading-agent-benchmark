#!/usr/bin/env python3
"""Adaptive AC monitor: starts at 30min, doubles on healthy, resets on error.

Checks DS (ac9wl_deepseek) and Terra (ac_luna_3wl_v5) WL progress, API health,
and supervisor liveness. On anomaly, attempts basic repair (reset account
cooldown, flush redis, restart supervisor). Writes state to /data/ac-monitor/.
"""
import json, os, subprocess, time, sys
from pathlib import Path
from datetime import datetime

STATE_DIR = Path("/data/ac-monitor")
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"
LOG_FILE = STATE_DIR / "monitor.log"
BASE_INTERVAL = 1800  # 30 min
MAX_INTERVAL = 14400  # 4 hours

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"interval": BASE_INTERVAL, "consecutive_healthy": 0}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))

def check_supervisor(name, pattern):
    try:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5)
        pids = [p for p in r.stdout.strip().split("\n") if p and "grep" not in p]
        return len(pids) > 0, len(pids)
    except:
        return False, 0

def check_wl_dates(log_dir, wls):
    results = {}
    for wl in wls:
        log = Path(log_dir) / f"{wl}.log"
        if not log.exists():
            results[wl] = None
            continue
        try:
            import re
            text = log.read_text(errors="replace")
            dates = re.findall(r"Current date (\d{4}-\d{2}-\d{2})", text)
            results[wl] = dates[-1] if dates else None
        except:
            results[wl] = None
    return results

def check_api_health():
    try:
        r = subprocess.run(
            ["sudo", "docker", "logs", "sub2api", "--since", "5m"],
            capture_output=True, text=True, timeout=10
        )
        out = r.stderr + r.stdout
        ok = out.count('"status_code": 200')
        err = sum(out.count(f'"status_code": {c}') for c in ("429", "502", "503"))
        return ok, err
    except:
        return 0, 0

def check_errors(log_dir, wls):
    errors = {}
    for wl in wls:
        log = Path(log_dir) / f"{wl}.log"
        if not log.exists():
            errors[wl] = 0
            continue
        try:
            # Only check last 1000 lines for recent errors
            text = log.read_text(errors="replace")
            recent = text[-50000:]  # last ~50KB
            count = recent.count("Traceback") + recent.count("Unexpected error")
            errors[wl] = count
        except:
            errors[wl] = 0
    return errors

def try_repair(issue):
    log(f"  attempting repair: {issue}")
    try:
        if "account" in issue or "503" in issue or "no available" in issue:
            subprocess.run([
                "sudo", "docker", "exec", "sub2api-postgres", "psql",
                "-U", "sub2api", "-d", "sub2api", "-c",
                "UPDATE accounts SET temp_unschedulable_until = NULL, "
                "temp_unschedulable_reason = NULL WHERE id = 1;"
            ], timeout=10, capture_output=True)
            subprocess.run([
                "sudo", "docker", "exec", "sub2api-redis", "redis-cli", "FLUSHALL"
            ], timeout=10, capture_output=True)
            log("  repair: reset account cooldown + redis flush")
        if "proxy" in issue or "connect" in issue:
            subprocess.run([
                "sudo", "iptables", "-t", "nat", "-A", "PREROUTING",
                "-p", "tcp", "-d", "172.18.0.1", "--dport", "7897",
                "-j", "DNAT", "--to-destination", "127.0.0.1:7897"
            ], timeout=5, capture_output=True)
            log("  repair: re-added iptables NAT rule")
    except Exception as e:
        log(f"  repair failed: {e}")

def run_check():
    issues = []
    
    # DS check
    ds_alive, ds_count = check_supervisor("DS", "run_deepseek_ac9")
    ds_dates = check_wl_dates(
        "/home/lxx/trade-agent-benchmark/AC-deepseek/results/ac9wl_deepseek/logs",
        ["wl1", "wl2", "wl3"]
    )
    ds_errors = check_errors(
        "/home/lxx/trade-agent-benchmark/AC-deepseek/results/ac9wl_deepseek/logs",
        ["wl1", "wl2", "wl3"]
    )
    if not ds_alive:
        issues.append("DS supervisor dead")
    
    # Terra check
    terra_alive, terra_count = check_supervisor("Terra", "run_ac_luna_3")
    terra_dates = check_wl_dates(
        "/home/lxx/trade-agent-benchmark/agent-framework/results/ac_luna_3wl_v5/logs",
        ["wl1", "wl2", "wl3"]
    )
    terra_errors = check_errors(
        "/home/lxx/trade-agent-benchmark/agent-framework/results/ac_luna_3wl_v5/logs",
        ["wl1", "wl2", "wl3"]
    )
    if not terra_alive:
        issues.append("Terra supervisor dead")
    
    # API health
    api_ok, api_err = check_api_health()
    if api_ok + api_err > 0:
        success_rate = api_ok / (api_ok + api_err)
        if success_rate < 0.7:
            issues.append(f"API success rate low: {success_rate:.0%} ({api_ok}ok/{api_err}err)")
    else:
        issues.append("No API traffic detected")
    
    # Report
    log(f"DS: alive={ds_alive} dates={ds_dates} errors={ds_errors}")
    log(f"Terra: alive={terra_alive} dates={terra_dates} errors={terra_errors}")
    log(f"API: {api_ok} ok, {api_err} errors")
    
    if issues:
        for issue in issues:
            log(f"ISSUE: {issue}")
            try_repair(issue)
        return False, issues
    else:
        log("All healthy")
        return True, []

def main():
    state = load_state()
    log(f"=== Monitor cycle (interval={state['interval']}s) ===")
    
    healthy, issues = run_check()
    
    if healthy:
        state["consecutive_healthy"] += 1
        state["interval"] = min(state["interval"] * 2, MAX_INTERVAL)
        log(f"Healthy ×{state['consecutive_healthy']}, next interval={state['interval']}s")
    else:
        state["consecutive_healthy"] = 0
        state["interval"] = BASE_INTERVAL
        log(f"Issues found, next interval reset to {BASE_INTERVAL}s")
    
    save_state(state)
    log(f"Next check in {state['interval']}s\n")

if __name__ == "__main__":
    main()
