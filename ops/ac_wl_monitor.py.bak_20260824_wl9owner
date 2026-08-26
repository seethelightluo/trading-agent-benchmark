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
        "wls": [f"wl{i}" for i in range(1, 10)],
        "proc": "run_ac_luna_3",
    },
}

# Terra run 自 2026-08-17 起按上游 fork 成多个 run 目录（避免并发写 run_state.json）。
# 每个目录只管理自己的 WL 子集（AC_LUNA_ONLY）；其余 WL 状态留在 v5 里。
# {run 目录名: 该目录拥有的 wl 编号}。sandbox 是共享的，无需区分。
TERRA_FORK_OWNERS = {
    "ac_luna_3wl_v5_oc": {4, 6, 8},      # opencode key2 (relay B :8788)
    "ac_luna_3wl_v5_plus": {5, 7, 9},    # ChatGPT Plus (relay A :8787)
}


def terra_wl_sources() -> dict[int, tuple[Path, Path]]:
    """每个 wl 编号 -> (run_state.json 路径, logs 目录)。优先 fork 目录，v5 兜底。"""
    base = WORKDIR / "agent-framework/results"
    out: dict[int, tuple[Path, Path]] = {}
    for wl in range(1, 10):
        out[wl] = (base / "ac_luna_3wl_v5" / "run_state.json",
                   base / "ac_luna_3wl_v5" / "logs")
    for sub, owns in TERRA_FORK_OWNERS.items():
        for wl in owns:
            out[wl] = (base / sub / "run_state.json", base / sub / "logs")
    return out


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


def _strip_ts(name: str) -> str:
    return re.sub(r"\.\d{8}T\d+.*\.json$", ".json", name)


def factor_stats(sandbox: Path, wl: str) -> dict:
    """Library stats from the audit contract; active count from the ensemble."""
    audit_path = sandbox / wl / "workspace/factor_library_audit.jsonl"
    kept = 0
    evicted: set[str] = set()
    conflict_sources: set[str] = set()
    audit_mtime = 0.0
    prev_cycle = None
    prev_kept = None
    if audit_path.exists():
        audit_mtime = audit_path.stat().st_mtime
        for line in audit_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                a = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Phase boundary: cycle counter restarted or the library was reseeded
            # (kept dropped sharply). Only the last phase is counted.
            boundary = (prev_cycle is not None and int(a["cycle"]) < prev_cycle) or (
                prev_kept is not None and prev_kept - int(a["kept"]) >= 10
            )
            if boundary:
                kept = 0
                evicted = set()
                conflict_sources = set()
            kept = int(a.get("kept", kept))
            for e in a.get("evicted") or []:
                evicted.add(_strip_ts(e))
            for c in a.get("conflicts") or []:
                src = c.get("source")
                if src:
                    conflict_sources.add(_strip_ts(src))
            prev_cycle = int(a["cycle"])
            prev_kept = kept
    evicted_other = evicted - conflict_sources
    active = None
    for ens_rel in ("workspace/factors/factor_ensemble.json", "workspace/factor_ensemble.json"):
        ens = read_json(sandbox / wl / ens_rel)
        if ens and isinstance(ens.get("selected_factors"), list):
            active = len(ens["selected_factors"])
            break
    factors_dir = sandbox / wl / "workspace/factors"
    live_disk = 0
    newest_file_mtime = 0.0
    if factors_dir.is_dir():
        for p in factors_dir.glob("*.json"):
            if p.name == "factor_ensemble.json" or p.name.endswith(".bak"):
                continue
            if re.search(r"\.\d{8}T\d+.*\.json$", p.name):
                continue
            live_disk += 1
            newest_file_mtime = max(newest_file_mtime, p.stat().st_mtime)
    return {
        "kept": kept,
        "active": active,
        "evicted": len(evicted),
        "evicted_conflict": len(conflict_sources),
        "evicted_other": len(evicted_other),
        "admitted": kept + len(evicted),
        "live_disk": live_disk,
        "audit_stale": audit_mtime > 0 and newest_file_mtime > audit_mtime,
    }


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
        wl_sources = terra_wl_sources() if group.startswith("Terra") else None
        print(f"== {group} (supervisor={'ALIVE' if alive else 'DEAD'}) ==")
        header = f"{'WL':<5}{'状态':<10}{'活跃':>4} {'库/累计':<10}{'淘汰(冲突/其他)':<16}{'NAV':>12}  {'窗口进度':<26}{'近1h推进':<14}"
        print(header)
        for wl in cfg["wls"]:
            status = "--"
            wl_rs = rs
            wl_log_dir = cfg["log_dir"]
            if wl_sources is not None:
                wl_no = int(wl[2:])
                st_path, lg_dir = wl_sources[wl_no]
                wl_rs = read_json(st_path) or rs
                wl_log_dir = lg_dir
            if wl_rs:
                wl_st = (wl_rs.get(wl) or {}).get("status")
                status = wl_st or "--"
            # fork 目录里没有 pause 标记的 paused_429 是 fork 时带入的旧状态，
            # 实际是排队等该目录 supervisor 启动（并发 1 顺位）
            if status == "paused_429" and wl_sources is not None:
                if not (st_path.parent / f"pause_{wl}_429").exists():
                    status = "queued"
            fs = factor_stats(cfg["sandbox"], wl)
            nav = latest_nav(cfg["sandbox"], wl)
            wp = window_progress(wl_log_dir, wl)
            win_no = wp["advances"] + 1 if wp["date"] else 0
            win_txt = f"{wp['date'] or '--'} w{win_no}/{max_cycles}"
            key = f"{group}:{wl}"
            hist.setdefault(key, [])
            one_h = advances_last_hour(hist, key, wp["advances"], now)
            hist[key].append([now, wp["advances"], wp["date"]])
            hist[key] = [e for e in hist[key] if e[0] >= now - HISTORY_WINDOW_SECONDS][-200:]
            active_txt = str(fs["active"]) if fs["active"] is not None else "--"
            elim_txt = f"{fs['evicted_conflict']}/{fs['evicted_other']}"
            kept_txt = f"{fs['kept']}*~{fs['live_disk']}" if fs["audit_stale"] else str(fs["kept"])
            print(f"{wl:<5}{status:<10}{active_txt:>4} {kept_txt}/{fs['admitted']:<8}{elim_txt:<16}{fmt_nav(nav):>12}  {win_txt:<26}{one_h:<14}")
        print()

    save_history(hist)
    print("说明: 活跃=ensemble selected_factors(<=10, 实际交易用); 库=审计 kept(<=30, 当前阶段), *~N 表示审计滞后于磁盘现存因子文件; 累计=库+当前阶段淘汰(去重); 淘汰: 冲突=rho>=0.5 质量低者出局, 其他=容量>30/重复ID; 近1h推进按快照差值估算, 首次显示 '--'.")


if __name__ == "__main__":
    main()
