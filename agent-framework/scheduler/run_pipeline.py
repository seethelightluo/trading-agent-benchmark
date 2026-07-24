#!/usr/bin/env python
"""
run_pipeline.py — 断电可恢复的 live 跑批运行器（9 WL × AC [+FM]）

设计：长任务（数小时~数天）必须抗断电、抗崩溃、可续跑。
- **逐 WL 调 AC**：每条世界线一次 `python main.py --session_id wlN --config run_config.yaml --resume`。
  AC 自带 cycle 级 --resume（从 logs 续跑），崩在哪条 cycle 重启即续。
- **失败重试**：AC 非零退出 → 等待退避后重新 --resume，直到 --max-retries。
- **状态持久化**：`results/run_state.json` 原子写入每条 WL 的完成状态 → 断电重启后自动跳过已完成项。
- **nohup/setsid 友好**：本脚本可被 `setsid nohup ... &` 拉起，脱离终端；单 WL 失败不中断后续 WL。

用法（live，需 LLM key 已配）：
    # 前台试跑 1 条 WL（小样本）
    python -m scheduler.run_pipeline --only 1 --mode ac --max-cycles 30
    # 后台全量（nohup，断电可恢复）
    setsid nohup .venv/bin/python -m scheduler.run_pipeline --mode both \\
        > results/run_pipeline.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # agent-framework/
AC_REPO = HERE / "AlphaCrafter" / "alphacrafter"
FM_REPO = HERE / "FactorMiner"
RESULTS = HERE / "results"
VENV_PY = Path("/home/lxx/trade-agent-benchmark/.venv/bin/python")

ONLINE_DIR = HERE.parent / "data-prepare" / "online-worldline"


def ac_env(cadence: int) -> dict:
    """AC 子进程环境：再平衡频次 + 继承当前 env（含 LLM key）。"""
    e = os.environ.copy()
    e["AC_CADENCE_DAYS"] = str(cadence)
    return e


def write_run_config(max_cycles: int, out: Path) -> Path:
    """生成 AC run 配置：复制 config.yaml，抬高 max_cycles（cadence-10 下每 WL ≈247 cycle）。"""
    src = AC_REPO / "config.yaml"
    text = src.read_text(encoding="utf-8")
    import re
    text = re.sub(r"max_cycles:\s*\d+", f"max_cycles: {max_cycles}", text)
    out.write_text(text, encoding="utf-8")
    return out


def load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)  # 原子写（防断电半写）


def run_ac_wl(wl: int, session: str, cfg: Path, cadence: int,
              retries: int, backoff: float) -> bool:
    """跑一条 WL 的 AC，带重试。返回是否最终成功。"""
    cmd = [str(VENV_PY), "main.py", "--session_id", session,
           "--config", str(cfg), "--resume"]
    for attempt in range(1, retries + 1):
        print(f"\n=== WL{wl} AC attempt {attempt}/{retries} ===\n  $ {' '.join(cmd)}  (cwd={AC_REPO})", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(AC_REPO), env=ac_env(cadence))
        dur = time.time() - t0
        if rc == 0:
            print(f"  ✅ WL{wl} AC 完成（{dur/3600:.1f}h）", flush=True)
            return True
        print(f"  ❌ WL{wl} AC rc={rc}（{dur/60:.1f}min），{backoff}s 后 --resume 重试", flush=True)
        time.sleep(backoff)
    return False


def _resample_panel(panel: Path, rule: str, out: Path) -> Path:
    """把日频长表 resample 到 rule（默认 10B=10 交易日），使 FM 的 quintile 回测
    每 step 再平衡 = 与 AC 同步的 cadence。OHLCV 聚合：open=first/high=max/low=min/close=last/volume/amount=sum。
    输出列对齐 FM loader：datetime, asset_id, OHLCV, amount。"""
    import pandas as pd
    df = pd.read_parquet(panel) if panel.suffix == ".parquet" else pd.read_csv(panel)
    df.columns = [c.lower() for c in df.columns]
    dc = "datetime" if "datetime" in df.columns else "date"
    df[dc] = pd.to_datetime(df[dc])
    agg = {c: f for c, f in {"open": "first", "high": "max", "low": "min",
                             "close": "last", "volume": "sum", "amount": "sum"}.items()
           if c in df.columns}
    parts = []
    for aid, g in df.groupby("asset_id"):
        r = (g.set_index(dc).sort_index().resample(rule).agg(agg)
             .dropna(subset=["close"]).reset_index().rename(columns={dc: "datetime"}))
        r["asset_id"] = aid
        parts.append(r[["datetime", "asset_id"] + list(agg)])
    out_df = pd.concat(parts, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    return out


def run_fm_wl(wl: int, panel: Path, retries: int, backoff: float,
              cadence_rule: str = "10B") -> bool:
    """跑一条 WL 的 FM：先 resample 到 cadence_rule（默认 10B→10 交易日再平衡，与 AC 对齐），
    再挖掘因子。FM 的再平衡=纯计算(0 LLM)；挖掘(LLM)频次由调度器/调用方决定，与此处的再平衡 cadence 无关。"""
    cfg = FM_REPO / "factorminer" / "configs" / "walkforward.yaml"
    env = {**os.environ, "PYTHONPATH": str(FM_REPO)}
    panel_res = RESULTS / f"WL{wl}_panel_{cadence_rule}.parquet"
    try:
        _resample_panel(panel, cadence_rule, panel_res)
        print(f"  FM panel resample → {cadence_rule}（{panel_res}）", flush=True)
    except Exception as e:
        print(f"  ⚠️ resample 失败({e})，用原日频 panel", flush=True)
        panel_res = panel
    script = ("from factorminer.cli import main; import sys; "
              "sys.argv=['factorminer','-c',%r,'mine','--data',%r]; main()"
              % (str(cfg), str(panel_res)))
    for attempt in range(1, retries + 1):
        print(f"\n=== WL{wl} FM attempt {attempt}/{retries} ===", flush=True)
        rc = subprocess.call([str(VENV_PY), "-c", script], cwd=str(FM_REPO), env=env)
        if rc == 0:
            return True
        print(f"  ❌ WL{wl} FM rc={rc}，{backoff}s 后重试", flush=True)
        time.sleep(backoff)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="逗号分隔 WL 号（默认 1-9 全跑）")
    ap.add_argument("--mode", default="ac", choices=["ac", "fm", "both"])
    ap.add_argument("--cadence", type=int, default=10, help="AC 再平衡频次（交易日/cycle）")
    ap.add_argument("--fm-cadence", default="10B", help="FM 再平衡 resample 规则（默认 10B=10交易日，与 AC 对齐）")
    ap.add_argument("--max-cycles", type=int, default=300, help="AC run config 的 max_cycles")
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--backoff", type=float, default=30.0, help="失败退避秒")
    ap.add_argument("--state", default=str(RESULTS / "run_state.json"))
    args = ap.parse_args()

    wls = [int(x) for x in args.only.split(",") if x.strip()] or list(range(1, 10))
    cfg = write_run_config(args.max_cycles, AC_REPO / "run_config.yaml")
    state_path = Path(args.state)
    state = load_state(state_path)
    print(f"运行器启动：WL={wls} mode={args.mode} cadence={args.cadence} "
          f"max_cycles={args.max_cycles} retries={args.max_retries}", flush=True)
    print(f"  AC run_config → {cfg}；状态 → {state_path}", flush=True)

    for wl in wls:
        key = f"wl{wl}"
        session = f"wl{wl}"
        panel = ONLINE_DIR / f"WL{wl}_full.parquet"
        st = state.get(key, {})
        if st.get("done"):
            print(f"\n⏭️  WL{wl} 已完成（state），跳过", flush=True)
            continue

        ok = True
        if args.mode in ("ac", "both") and not st.get("ac_done"):
            ok_ac = run_ac_wl(wl, session, cfg, args.cadence, args.max_retries, args.backoff)
            st["ac_done"] = ok_ac
            state[key] = st
            save_state(state_path, state)
            ok = ok and ok_ac
        if args.mode in ("fm", "both") and ok and not st.get("fm_done") and panel.exists():
            ok_fm = run_fm_wl(wl, panel, args.max_retries, args.backoff, args.fm_cadence)
            st["fm_done"] = ok_fm
            state[key] = st
            save_state(state_path, state)
            ok = ok and ok_fm

        st["done"] = ok
        st["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state[key] = st
        save_state(state_path, state)
        print(f"\n{'✅' if ok else '⚠️'} WL{wl} 结束（done={ok}）", flush=True)

    done = sum(1 for v in state.values() if v.get("done"))
    print(f"\n========== 全部结束：{done}/{len(wls)} WL 完成 ==========", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
