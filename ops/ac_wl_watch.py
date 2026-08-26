#!/usr/bin/env python3
"""AC WL progress watcher: compact snapshot + anomaly flags every N minutes.

Logs to ops/logs/ac_wl_watch.log.  Anomalies flagged:
- running WL with no step advance within STALL_MIN minutes
- new 'stale AC online account contract' lines appearing in a WL log
- NAV/date file unreadable
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/lxx/trade-agent-benchmark")
DS_LOG = ROOT / "AC-deepseek" / "results" / "ac9wl_deepseek" / "logs"
# Terra 自 2026-08-17 按 LLM 上游 fork 成多个 run 目录；每个 WL 的日志和
# pause 标记都在它所属目录里（supervisor 只看自己 RUN_DIR 的标记）。
TERRA_RES_BASE = ROOT / "agent-framework" / "results"
_TERRA_FORK = {"ac_luna_3wl_v5_oc": (4, 6, 8, 9), "ac_luna_3wl_v5_plus": (5, 7)}
TERRA_WL_DIR: dict[str, Path] = {
    f"wl{i}": TERRA_RES_BASE / "ac_luna_3wl_v5" for i in range(1, 10)
}
for _sub, _owns in _TERRA_FORK.items():
    for _wl in _owns:
        TERRA_WL_DIR[f"wl{_wl}"] = TERRA_RES_BASE / _sub
OUT = ROOT / "ops" / "logs" / "ac_wl_watch.log"
SLEEP = int(os.environ.get("AC_WATCH_INTERVAL", "300"))
STALL_MIN = int(os.environ.get("AC_WATCH_STALL_MIN", "60"))
AUTO_PAUSE = os.environ.get("AC_WATCH_AUTO_PAUSE", "1").lower() in {"1", "true", "yes"}
STALL_LOG = ROOT / "ops" / "logs" / "ac_wl_stalls.log"
STATE_FILE = ROOT / "ops" / "logs" / "ac_wl_watch_state.json"
DS_RESULTS = ROOT / "AC-deepseek" / "results" / "ac9wl_deepseek"
TERRA_RESULTS = ROOT / "agent-framework" / "results" / "ac_luna_3wl_v5"
PAUSE_DIRS = {"ds": DS_RESULTS, "terra": TERRA_RESULTS}

def terra_log_dir(wl: str) -> Path:
    return TERRA_WL_DIR[wl] / "logs"


def terra_pause_dir(wl: str) -> Path:
    return TERRA_WL_DIR[wl]


ADV_RE = re.compile(r"Advanced 10 trading days")
DATE_RE = re.compile(r"Current date (\d{4}-\d{2}-\d{2})")
STALE_RE = re.compile(r"stale AC online account contract")

FAMILIES = [("terra", range(1, 10)), ("ds", range(1, 10))]
last_seen: dict[str, tuple[float, int, int, bool]] = {}  # key -> (ts, advances, stale, running)


def load_last_seen() -> dict[str, tuple[float, int, int, bool]]:
    """Restore the stall clock across watcher restarts so a restart does not
    reset the 'no advance' window for running worldlines."""
    try:
        raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    out: dict[str, tuple[float, int, int, bool]] = {}
    for key, value in raw.items():
        try:
            ts, adv, stale, running = value
            out[key] = (float(ts), int(adv), int(stale), bool(running))
        except (TypeError, ValueError):
            continue
    return out


def save_last_seen() -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(
            json.dumps(last_seen, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
    except OSError:
        pass


def proc_state() -> dict:
    """Return {(family, wl): pid} for live main.py wl processes."""
    out = subprocess.run(
        ["ps", "-eo", "pid,cmd"], capture_output=True, text=True, timeout=30
    ).stdout
    res: dict[tuple[str, str], str] = {}
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        cmd = parts[1]
        if "main.py wl" not in cmd or "bash" in cmd or "rg " in cmd:
            continue
        m = re.search(r"main\.py (wl\d+)", cmd)
        if not m:
            continue
        if "AC-deepseek" in cmd:
            fam = "ds"
        elif "agent-framework" in cmd:
            fam = "terra"
        else:
            continue
        res[(fam, m.group(1))] = parts[0]
    return res


def wl_metrics(log_dir: Path, wl: str) -> dict:
    log = log_dir / f"{wl}.log"
    if not log.exists():
        return {"date": None, "adv": 0, "stale": 0}
    text = log.read_text(encoding="utf-8", errors="replace")
    m = DATE_RE.findall(text)
    date = m[-1] if m else None
    return {
        "date": date,
        "adv": len(ADV_RE.findall(text)),
        "stale": len(STALE_RE.findall(text)),
    }


def nav_rebal(wl: str, fam: str) -> tuple[float | None, int | None]:
    base = ROOT / ("AC-deepseek" if fam == "ds" else "agent-framework")
    sandbox = base / "AlphaCrafter" / "alphacrafter" / "sandbox"
    try:
        a = json.loads((sandbox / wl / "persistent" / "account.json").read_text())
        return round(float(a.get("net_assets", 0.0)), 2), len(a.get("rebalance_history", []))
    except Exception:
        return None, None


def pause_wl(fam: str, wl: str, pid: str, reason: str) -> bool:
    """Create the scheduler pause marker, SIGINT the process, and log a stall
    record for manual debugging.  Returns True when this call paused the WL."""
    base_dir = terra_pause_dir(wl) if fam == "terra" else PAUSE_DIRS[fam]
    marker = base_dir / f"pause_{wl}_429"
    if marker.exists():
        return False
    try:
        marker.write_text(
            f"auto-paused by ac_wl_watch at {datetime.now().isoformat(timespec='seconds')}\n{reason}\n"
        )
    except OSError as exc:
        print(f"WARN cannot write pause marker {marker}: {exc}", flush=True)
        return False
    try:
        os.kill(int(pid), 2)  # SIGINT -> graceful agent stop
    except (OSError, ValueError) as exc:
        print(f"WARN cannot SIGINT pid {pid}: {exc}", flush=True)
    STALL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with STALL_LOG.open("a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now().isoformat(timespec='seconds')} | {fam} {wl} "
            f"pid={pid} | marker={marker} | {reason}\n"
        )
    return True


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    last_seen.update(load_last_seen())
    save_last_seen()
    # one-shot stamp at startup
    with OUT.open("a", encoding="utf-8") as f:
        f.write(f"=== watcher started {datetime.now().isoformat(timespec='seconds')} ===\n")
    while True:
        procs = proc_state()
        now = time.time()
        lines = []
        flags = []
        for fam, wls in FAMILIES:
            parts = []
            for i in wls:
                wl = f"wl{i}"
                log_dir = terra_log_dir(wl) if fam == "terra" else DS_LOG
                met = wl_metrics(log_dir, wl)
                nav, rebal = nav_rebal(wl, fam)
                running = (fam, wl) in procs
                key = f"{fam}-{wl}"
                prev = last_seen.get(key)
                if prev is not None and running:
                    ts0, adv0, stale0, prev_running = prev
                    if prev_running and met["adv"] == adv0 and now - ts0 > STALL_MIN * 60:
                        flags.append(f"{key}:STALL(date={met['date']})")
                        if AUTO_PAUSE and not (PAUSE_DIRS[fam] / f"pause_{wl}_429").exists():
                            paused = pause_wl(
                                fam, wl, procs[(fam, wl)],
                                f"stalled >= {STALL_MIN}min (adv={adv0}, date={met['date']})",
                            )
                            flags.append(f"{key}:PAUSED({paused})")
                    if met["stale"] > stale0:
                        flags.append(f"{key}:NEW_STALE({met['stale'] - stale0})")
                if prev is None or met["adv"] != prev[1] or running != prev[3]:
                    last_seen[key] = (now, met["adv"], met["stale"], running)
                    save_last_seen()
                win = met["adv"] + 1 if met["adv"] else 0
                st = "run" if running else "queued"
                nav_t = f"{nav:,}" if nav is not None else "NA"
                parts.append(
                    f"{wl}[{st}](w{win}) {met['date'] or '--'} nav={nav_t} r={rebal}"
                )
            lines.append(f"{fam}: " + " | ".join(parts))
        stamp = datetime.now().isoformat(timespec="seconds")
        flag_txt = (" FLAGS: " + " ".join(flags)) if flags else ""
        line = stamp + " | " + " | ".join(lines) + flag_txt
        with OUT.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        time.sleep(SLEEP)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pass
