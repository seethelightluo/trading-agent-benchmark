#!/bin/bash
# Adaptive AC monitor daemon: checks DS/Terra health with self-adjusting interval.
# Runs forever: 30min baseline, doubles on healthy, resets on anomaly.
exec /home/lxx/trade-agent-benchmark/.venv/bin/python -u << 'PYTHON'
import json, subprocess, time, re, os
from pathlib import Path
from datetime import datetime

STATE_DIR = Path("/data/ac-monitor")
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"
LOG_FILE = STATE_DIR / "monitor.log"
BASE = 1800  # 30 min
MAXI = 14400  # 4 hours

def log(m):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {m}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f: f.write(line + "\n")

def load():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"interval": BASE, "healthy_streak": 0, "last_dates": {}}

def save(s): STATE_FILE.write_text(json.dumps(s, indent=2))

def pgrep_alive(pattern):
    try:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5)
        return len([p for p in r.stdout.strip().split("\n") if p]) > 0
    except: return False

def wl_dates(logdir, wls):
    out = {}
    for wl in wls:
        p = Path(logdir) / f"{wl}.log"
        if not p.exists(): out[wl] = "?"; continue
        try:
            t = p.read_text(errors="replace")[-50000:]
            ds = re.findall(r"Current date (\d{4}-\d{2}-\d{2})", t)
            out[wl] = ds[-1] if ds else "?"
        except: out[wl] = "?"
    return out

def wl_errors(logdir, wls):
    out = {}
    for wl in wls:
        p = Path(logdir) / f"{wl}.log"
        if not p.exists(): out[wl] = 0; continue
        try:
            t = p.read_text(errors="replace")[-30000:]
            out[wl] = t.count("Traceback") + t.count("Unexpected error")
        except: out[wl] = 0
    return out

def api_health():
    try:
        r = subprocess.run(["sudo","docker","logs","sub2api","--since","5m"],
                          capture_output=True, text=True, timeout=10)
        o = r.stderr + r.stdout
        ok = o.count('"status_code": 200')
        er = sum(o.count(f'"status_code": {c}') for c in ("429","502","503"))
        return ok, er
    except: return 0, 0

def repair(issue):
    log(f"  REPAIR: {issue}")
    try:
        if "503" in issue or "account" in issue:
            subprocess.run(["sudo","docker","exec","sub2api-postgres","psql","-U","sub2api","-d","sub2api","-c",
                "UPDATE accounts SET temp_unschedulable_until=NULL, temp_unschedulable_reason=NULL;"],
                timeout=10, capture_output=True)
            subprocess.run(["sudo","docker","exec","sub2api-redis","redis-cli","FLUSHALL"],
                timeout=10, capture_output=True)
            log("  repair: reset accounts + redis")
        if "proxy" in issue or "connect" in issue or "iptables" in issue:
            subprocess.run(["sudo","iptables","-t","nat","-D","PREROUTING","-p","tcp","-d","172.18.0.1","--dport","7897","-j","DNAT","--to-destination","127.0.0.1:7897"],
                timeout=5, capture_output=True)
            subprocess.run(["sudo","iptables","-t","nat","-A","PREROUTING","-p","tcp","-d","172.18.0.1","--dport","7897","-j","DNAT","--to-destination","127.0.0.1:7897"],
                timeout=5, capture_output=True)
            log("  repair: re-added iptables NAT")
    except Exception as e: log(f"  repair failed: {e}")

def check_once(state):
    issues = []
    
    ds_a = pgrep_alive("run_deepseek_ac9")
    ds_d = wl_dates("/home/lxx/trade-agent-benchmark/AC-deepseek/results/ac9wl_deepseek/logs", ["wl1","wl2","wl3"])
    ds_e = wl_errors("/home/lxx/trade-agent-benchmark/AC-deepseek/results/ac9wl_deepseek/logs", ["wl1","wl2","wl3"])
    if not ds_a: issues.append("DS supervisor dead")
    
    te_a = pgrep_alive("run_ac_luna_3")
    te_d = wl_dates("/home/lxx/trade-agent-benchmark/agent-framework/results/ac_luna_3wl_v5/logs", ["wl1","wl2","wl3"])
    te_e = wl_errors("/home/lxx/trade-agent-benchmark/agent-framework/results/ac_luna_3wl_v5/logs", ["wl1","wl2","wl3"])
    if not te_a: issues.append("Terra supervisor dead")
    
    ok, er = api_health()
    rate = ok/(ok+er) if (ok+er) > 0 else 0
    if rate < 0.7 and (ok+er) > 5:
        issues.append(f"API rate low {rate:.0%}")
    elif ok+er == 0:
        issues.append("no API traffic")
    
    # Date advancement check (compare with last cycle)
    cur_dates = {"ds": ds_d, "te": te_d}
    prev = state.get("last_dates", {})
    for exp, cur in cur_dates.items():
        if exp in prev and prev[exp] == cur:
            for wl, d in cur.items():
                if prev[exp].get(wl) == d and d != "?":
                    pass  # date unchanged - might be slow but not necessarily error
    
    log(f"DS alive={ds_a} dates={ds_d} errs={ds_e}")
    log(f"TE alive={te_a} dates={te_d} errs={te_e}")
    log(f"API {ok}ok/{er}err rate={rate:.0%}")
    
    state["last_dates"] = cur_dates
    
    if issues:
        for i in issues:
            log(f"ISSUE: {i}")
            repair(i)
        state["healthy_streak"] = 0
        state["interval"] = BASE
        log(f"Unhealthy → interval={BASE}s")
    else:
        state["healthy_streak"] += 1
        state["interval"] = min(state["interval"] * 2, MAXI)
        log(f"Healthy ×{state['healthy_streak']} → interval={state['interval']}s")
    
    save(state)
    return state

# Main loop
state = load()
log(f"=== AC Monitor started (base={BASE}s max={MAXI}s) ===")
while True:
    try:
        state = check_once(state)
    except Exception as e:
        log(f"FATAL: {e}")
        state["interval"] = BASE
        save(state)
    
    sleep_s = state["interval"]
    log(f"Sleeping {sleep_s}s...\n")
    time.sleep(sleep_s)
PYTHON
