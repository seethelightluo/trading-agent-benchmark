#!/usr/bin/env python3
"""AC global WL monitor: factor library, NAV, window progress, 1h advances.

Usage: python3 ops/ac_wl_monitor.py
State (baseline for the "1h advances" estimate) is kept in /data/ac-monitor/.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

WORKDIR = Path("/home/lxx/trade-agent-benchmark")
STATE_DIR = Path("/data/ac-monitor")
STATE_FILE = STATE_DIR / "wl_progress_history.json"
HISTORY_WINDOW_SECONDS = 24 * 3600

GROUPS = {
    "DS (deepseek)": {
        "run_state": WORKDIR / "AC-deepseek/results/ac9wl_deepseek/run_state.json",
        "log_dir": WORKDIR / "AC-deepseek/results/ac9wl_deepseek/logs",
        "sandbox": WORKDIR / "AC-deepseek/AlphaCrafter/alphacrafter/sandbox",
        "wls": [f"wl{i}" for i in range(1, 10)],
        "proc": "run_deepseek_ac9",
    },
    "Terra (luna)": {
        "run_state": WORKDIR / "agent-framework/results/ac_luna_3wl_v5/run_state.json",
        "log_dir": WORKDIR / "agent-framework/results/ac_luna_3wl_v5/logs",
        "sandbox": WORKDIR / "agent-framework/AlphaCrafter/alphacrafter/sandbox",
        "wls": [f"wl{i}" for i in range(1, 4)],
        "proc": "run_ac_luna_3",
    },
}


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def proc_alive(pattern: str) -> bool:
    try:
        import subprocess

        r = subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5
        )
        return bool(r.stdout.strip())
    except Exception:
        return False


def last_match(text: str, pattern: str) -> str | None:
    m = list(re.finditer(pattern, text))
    return m[-1].group(1) if m else None


def factor_stats(sandbox: Path, wl: str) -> dict:
    audit_path = sandbox / wl / "workspace/factor_library_audit.jsonl"
    kept = 0
    evicted: set[str] = set()
    if audit_path.exists():
        for line in audit_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                a = json.loads(line)
            except json.JSONDecodeError:
                continue
            kept = int(a.get("kept", kept))
            for e in a.get("evicted") or []:
                e = re.sub(r"\.\d{8}T\d+.*\.json$", ".json", e)
                evicted.add(e)
    return {"kept": kept, "evicted": len(evicted), "admitted": kept + len(evicted)}


def latest_nav(sandbox: Path, wl: str) -> float | None:
    acct = read_json(sandbox / wl / "persistent/account.json")
    if not acct:
        return None
    return acct.get("net_assets")


def window_progress(log_dir: Path, wl: str) -> dict:
    log = log_dir / f"{wl}.log"
    if not log.exists():
        return {"date": None, "advances": 0}
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"date": None, "advances": 0}
    date = last_match(text, r"Current date (\d{4}-\d{2}-\d{2})")
    advances = len(re.findall(r"Advanced 10 trading days", text))
    return {"date": date, "advances": advances}


def load_history() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def save_history(hist: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(hist, ensure_ascii=False), encoding="utf-8")


def advances_last_hour(hist: dict, key: str, advances: int, now: float) -> str:
    entries = hist.get(key) or []
    now_iso = datetime.fromtimestamp(now).isoformat(timespec="seconds")
    if not entries:
        return "--"
    cutoff = now - 3600
    in_window = [e for e in entries if e[0] >= cutoff]
    if in_window:
        base_ts, base_adv = in_window[0][0], in_window[0][1]
    else:
        base_ts, base_adv = entries[-1][0], entries[-1][1]
    delta = max(0, advances - base_adv)
    span_min = max(0, int((now - base_ts) // 60))
    if span_min >= 55:
        return f"+{delta} ({span_min}min)"
    return f"+{delta}"


def fmt_nav(nav: float | None) -> str:
    return f"{nav:,.0f}" if nav is not None else "--"


def main() -> None:
    now = time.time()
    hist = load_history()
    print(f"AC global monitor @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for group, cfg in GROUPS.items():
        alive = proc_alive(cfg["proc"])
        rs = read_json(cfg["run_state"])
        max_cycles = rs.get("online_max_cycles", 300) if rs else 300
        print(f"== {group} (supervisor={'ALIVE' if alive else 'DEAD'}) ==")
        header = f"{'WL':<5}{'状态':<10}{'因子 当前/累计/替换':<22}{'NAV':>12}  {'窗口进度':<26}{'近1h推进':<14}"
        print(header)
        for wl in cfg["wls"]:
            status = "--"
            if rs:
                wl_st = (rs.get(wl) or {}).get("status")
                status = wl_st or "--"
            fs = factor_stats(cfg["sandbox"], wl)
            nav = latest_nav(cfg["sandbox"], wl)
            wp = window_progress(cfg["log_dir"], wl)
            win_no = wp["advances"] + 1 if wp["date"] else 0
            win_txt = f"{wp['date'] or '--'} w{win_no}/{max_cycles}"
            key = f"{group}:{wl}"
            hist.setdefault(key, [])
            one_h = advances_last_hour(hist, key, wp["advances"], now)
            hist[key].append([now, wp["advances"], wp["date"]])
            hist[key] = [e for e in hist[key] if e[0] >= now - HISTORY_WINDOW_SECONDS][-200:]
            facts = f"{fs['kept']}/{fs['admitted']}/{fs['evicted']}"
            print(f"{wl:<5}{status:<10}{facts:<22}{fmt_nav(nav):>12}  {win_txt:<26}{one_h:<14}")
        print()

    save_history(hist)
    print("说明: 近1h推进按本次与上次运行快照差值估算, 首次运行显示 '--'; 因子=审计 kept; 累计=kept+历史替换; 替换=去重 evicted.")


if __name__ == "__main__":
    main()
