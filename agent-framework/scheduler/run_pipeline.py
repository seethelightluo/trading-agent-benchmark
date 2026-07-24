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


class EscalatingBackoff:
    """配额 5h 刷新模型下的重试退避：立即→1分→10分→1小时，封顶 1 小时。
    失败累加索引；**任一成功立即重置到立即**。共享于全 pipeline（一条 WL 成功后，下条 WL 的失败重新从立即开始）。"""
    SCHEDULE = [0, 60, 600, 3600]  # 秒：立即 / 1分 / 10分 / 1小时

    def __init__(self):
        self.idx = 0

    def on_fail(self) -> float:
        wait = self.SCHEDULE[min(self.idx, len(self.SCHEDULE) - 1)]
        self.idx += 1
        return wait

    def on_success(self):
        self.idx = 0


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
              bk: EscalatingBackoff, max_attempts: int = 0) -> bool:
    """跑一条 WL 的 AC，失败按递增退避(立即→1分→10分→1h)重试（配额5h刷新模型）。
    AC --resume 从最近 cycle 续跑，不丢进度；成功则重置退避。
    max_attempts=0 无限重试（全量默认）；>0 限次（冒烟用，遇真 bug 不死循环）。"""
    cmd = [str(VENV_PY), "main.py", "--session_id", session,
           "--config", str(cfg), "--resume"]
    attempt = 0
    while True:
        attempt += 1
        print(f"\n=== WL{wl} AC attempt {attempt}（退避档{min(bk.idx, len(bk.SCHEDULE)-1)+1}）===\n  $ {' '.join(cmd)}  (cwd={AC_REPO})", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(AC_REPO), env=ac_env(cadence))
        dur = time.time() - t0
        if rc == 0:
            bk.on_success()
            print(f"  ✅ WL{wl} AC 完成（{dur/3600:.1f}h），退避已重置", flush=True)
            return True
        if max_attempts and attempt >= max_attempts:
            print(f"  ⛔ WL{wl} AC 达 max_attempts={max_attempts}，放弃（冒烟模式）", flush=True)
            return False
        wait = bk.on_fail()
        print(f"  ❌ WL{wl} AC rc={rc}（{dur/60:.1f}min）；{wait}s 后 --resume 重试（配额5h刷新，递增退避）", flush=True)
        if wait:
            time.sleep(wait)


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


def run_fm_wl(wl: int, panel: Path, bk: EscalatingBackoff,
              cadence_rule: str = "10B", live: bool = True, max_attempts: int = 0) -> bool:
    """跑一条 WL 的 FM：先 resample 到 cadence_rule（默认 10B→10 交易日再平衡，与 AC 对齐），
    再挖掘因子。live=True 用 fm_live.yaml（OpenAI兼容/glm-5.2），False 用 fm_mock_real.yaml。
    失败按递增退避(立即→1分→10分→1h)重试；成功重置。max_attempts=0 无限，>0 限次（冒烟）。"""
    cfg = FM_REPO / "factorminer" / "configs" / ("fm_live.yaml" if live else "fm_mock_real.yaml")
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
    attempt = 0
    while True:
        attempt += 1
        print(f"\n=== WL{wl} FM attempt {attempt}（退避档{min(bk.idx, len(bk.SCHEDULE)-1)+1}，cfg={cfg.name}）===", flush=True)
        rc = subprocess.call([str(VENV_PY), "-c", script], cwd=str(FM_REPO), env=env)
        if rc == 0:
            bk.on_success()
            print(f"  ✅ WL{wl} FM 完成，退避已重置", flush=True)
            return True
        if max_attempts and attempt >= max_attempts:
            print(f"  ⛔ WL{wl} FM 达 max_attempts={max_attempts}，放弃（冒烟模式）", flush=True)
            return False
        wait = bk.on_fail()
        print(f"  ❌ WL{wl} FM rc={rc}；{wait}s 后重试（配额5h刷新，递增退避）", flush=True)
        if wait:
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="逗号分隔 WL 号（默认 1-9 全跑）")
    ap.add_argument("--mode", default="ac", choices=["ac", "fm", "both"])
    ap.add_argument("--cadence", type=int, default=10, help="AC 再平衡频次（交易日/cycle）")
    ap.add_argument("--fm-cadence", default="10B", help="FM 再平衡 resample 规则（默认 10B=10交易日，与 AC 对齐）")
    ap.add_argument("--max-cycles", type=int, default=300, help="AC run config 的 max_cycles")
    ap.add_argument("--fm-mock", action="store_true", help="FM 用 mock provider（默认 live：fm_live.yaml/glm-5.2）")
    ap.add_argument("--max-attempts", type=int, default=0, help="每 WL 最大尝试次数（0=无限重试，全量默认；冒烟设 3 遇真bug不死循环）")
    ap.add_argument("--state", default=str(RESULTS / "run_state.json"))
    args = ap.parse_args()

    wls = [int(x) for x in args.only.split(",") if x.strip()] or list(range(1, 10))
    cfg = write_run_config(args.max_cycles, AC_REPO / "run_config.yaml")
    state_path = Path(args.state)
    state = load_state(state_path)
    bk = EscalatingBackoff()  # 全 pipeline 共享：立即→1分→10分→1h，成功重置
    print(f"运行器启动：WL={wls} mode={args.mode} cadence={args.cadence} "
          f"max_cycles={args.max_cycles} fm={'mock' if args.fm_mock else 'live'}", flush=True)
    print(f"  退避档(秒)：{EscalatingBackoff.SCHEDULE}（配额5h刷新模型，成功重置）", flush=True)
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
            ok_ac = run_ac_wl(wl, session, cfg, args.cadence, bk, args.max_attempts)
            st["ac_done"] = ok_ac
            state[key] = st
            save_state(state_path, state)
            ok = ok and ok_ac
        if args.mode in ("fm", "both") and ok and not st.get("fm_done") and panel.exists():
            ok_fm = run_fm_wl(wl, panel, bk, args.fm_cadence, not args.fm_mock, args.max_attempts)
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
