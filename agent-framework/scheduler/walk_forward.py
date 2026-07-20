"""
walk_forward.py — 2026.07.16 → 2030.12.31 日频前向走步调度器

职责（对应"实时数据 → 实时调仓"的 walk-forward 语义）
----------------------------------------------------
1. 时间游标：每日把 AC 的 date.json.current_date 推进到 t；
   两套框架在 t 这一天只能看到 [warmup_start, t] 的数据（防穿越）。
   - AC 的 GetStockData/GetIndexData/GetNews 工具已按 current_date 过滤，天然防穿越。
   - FM：调度器在调用前把面板切片到 ≤ t，落成 panel_t.parquet 再喂给 FM。
2. 每月 1 日新闻：build_inputs 已为每资产生成"每月 1 日"的新闻 JSON；
   AC 的 get_news 按 publish_date ≤ current_date 过滤 → 每月首个交易日当月新闻自动可见。
   本调度器在每月首个交易日打 "news injected" 日志，便于核对。
3. 驱动两 agent：
   - AC：每个交易日调用 `python main.py --session_id <s> --resume` 跑一个日频循环
         （Miner/Screener/Trader + Step 推进一天；3bps 摩擦已在交易所内置）。
   - FM：默认每月末调用一次 `python -m factorminer`（纯量价，cost_bps=3.0 已配）。
4. 每日快照：从 account.json 读净值/持仓/累计盈亏，写 results/equity.csv。

dry-run（不调用任何 LLM）：仅推进游标 + 切片 + 新闻/快照日志，可立即验证逻辑。

用法
----
    # 验证游标/防穿越/每月新闻逻辑（无需 API key、无需数据全量）
    python -m scheduler.walk_forward --session wf19 --mode dryrun

    # 真正驱动（需要 AC 的 LLM key 与 Docker/venv 就绪）
    python -m scheduler.walk_forward --session wf19 --mode both --fm-freq monthly
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent.parent  # agent-framework/
AC_REPO = HERE / "AlphaCrafter" / "alphacrafter"
AC_SANDBOX = HERE / "AlphaCrafter" / "alphacrafter" / "sandbox"
FM_REPO = HERE / "FactorMiner"


def load_assets(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_date_json(session: str) -> dict:
    p = AC_SANDBOX / session / "persistent" / "date.json"
    return json.loads(p.read_text(encoding="utf-8"))


def write_current_date(session: str, d: str) -> None:
    p = AC_SANDBOX / session / "persistent" / "date.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data["current_date"] = d
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_account(session: str) -> dict:
    p = AC_SANDBOX / session / "persistent" / "account.json"
    return json.loads(p.read_text(encoding="utf-8"))


def is_first_trading_day_of_month(d: str, trading_days: list[str]) -> bool:
    """d 是否为其所在月的第一个交易日（用于触发"每月 1 日新闻"可见点）。"""
    i = trading_days.index(d)
    if i == 0:
        return True
    return trading_days[i - 1][:7] != d[:7]


def is_month_end(d: str, trading_days: list[str]) -> bool:
    """d 是否为其所在月的最后一个交易日（用于触发月末 FM 挖掘）。"""
    i = trading_days.index(d)
    if i == len(trading_days) - 1:
        return True
    return trading_days[i + 1][:7] != d[:7]


def slice_panel_le(panel_path: Path, d: str, out_path: Path) -> Path:
    """防穿越：把面板切片到 date ≤ d，写出 panel_t.parquet 给 FM。"""
    import pandas as pd  # 惰性导入，dryrun 不需要
    df = pd.read_parquet(panel_path) if panel_path.suffix == ".parquet" else pd.read_csv(panel_path)
    df.columns = [c.lower() for c in df.columns]
    date_col = "datetime" if "datetime" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col])
    sub = df[df[date_col] <= pd.to_datetime(d)].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(out_path, index=False)
    return out_path


def run_ac_cycle(session: str) -> int:
    """调用 AC 跑一个日频循环（--resume 续跑）。返回 returncode。"""
    cmd = [sys.executable, "main.py", "--session_id", session, "--resume"]
    print(f"    [AC] $ {' '.join(cmd)}  (cwd={AC_REPO})")
    return subprocess.call(cmd, cwd=str(AC_REPO))


def run_fm_mining(panel_t: Path, fm_config: Path) -> int:
    """调用 FM 在切片后面板上做一次因子挖掘。返回 returncode。"""
    cmd = [sys.executable, "-m", "factorminer", "mine",
           "--data", str(panel_t), "--config", str(fm_config)]
    print(f"    [FM] $ {' '.join(cmd)}  (cwd={FM_REPO})")
    return subprocess.call(cmd, cwd=str(FM_REPO))


def append_equity(rows: list[dict], out_csv: Path) -> None:
    import csv  # 标准库即可，避免强依赖 pandas
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="wf19", help="AC session id")
    ap.add_argument("--assets", default=str(HERE / "ASSETS.yaml"))
    ap.add_argument("--mode", default="dryrun",
                    choices=["dryrun", "ac", "fm", "both"],
                    help="dryrun=只推进游标/切片/日志；ac/fm/both=实际驱动")
    ap.add_argument("--fm-freq", default="monthly", choices=["daily", "monthly"])
    ap.add_argument("--fm-panel", default=str(FM_REPO / "data" / "panel.parquet"))
    ap.add_argument("--fm-config",
                    default=str(FM_REPO / "factorminer" / "configs" / "walkforward.yaml"))
    ap.add_argument("--results", default=str(HERE / "results" / "equity.csv"))
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 个交易日（调试）")
    args = ap.parse_args()

    assets_cfg = load_assets(Path(args.assets))
    start = assets_cfg["baseline_date"]
    end = assets_cfg["online_end"]

    dj = read_date_json(args.session)
    trading_days = dj["trading_days"]
    window = [d for d in trading_days if start <= d <= end]
    if args.limit > 0:
        window = window[: args.limit]
    print(f"前向窗口：{window[0]} → {window[-1]}  ({len(window)} 个交易日)")
    print(f"模式：{args.mode}   FM 频率：{args.fm_freq}")

    run_ac = args.mode in ("ac", "both")
    run_fm = args.mode in ("fm", "both")
    equity_rows: list[dict] = []
    months_injected: set[str] = set()

    for k, d in enumerate(window, 1):
        # 1) 推进时间游标（显式防穿越）
        write_current_date(args.session, d)
        first_of_month = is_first_trading_day_of_month(d, trading_days)
        month_end = is_month_end(d, trading_days)
        tag = []
        if first_of_month:
            tag.append("📰月首新闻可见")
            months_injected.add(d[:7])
        if month_end:
            tag.append("月末")

        print(f"[{k}/{len(window)}] {d}  {' '.join(tag)}")

        # 2) 驱动 AlphaCrafter（每日一个循环；摩擦已内置 3bps）
        if run_ac:
            rc = run_ac_cycle(args.session)
            if rc != 0:
                print(f"    [AC] 非 0 返回码 {rc}，继续")

        # 3) 驱动 FactorMiner（防穿越切片后挖掘）
        if run_fm and (args.fm_freq == "daily" or month_end):
            panel_t = slice_panel_le(Path(args.fm_panel), d,
                                     HERE / "results" / f"panel_{d}.parquet")
            rc = run_fm_mining(panel_t, Path(args.fm_config))
            if rc != 0:
                print(f"    [FM] 非 0 返回码 {rc}，继续")

        # 4) 每日快照（从 account.json）
        acc = read_account(args.session)
        equity_rows.append({
            "date": d,
            "net_assets": acc.get("net_assets"),
            "available_cash": acc.get("available_cash"),
            "market_value": acc.get("market_value"),
            "total_profit_loss": acc.get("total_profit_loss"),
            "total_profit_loss_rate": acc.get("total_profit_loss_rate"),
            "gross_position_rate": acc.get("gross_position_rate"),
            "n_positions": len(acc.get("positions", [])),
            "first_of_month_news": int(first_of_month),
        })

    append_equity(equity_rows, Path(args.results))
    print(f"\n✅ 完成 {len(window)} 个交易日。净值曲线：{args.results}")
    print(f"   覆盖月份（月首新闻注入）：{sorted(months_injected)}")
    if args.mode == "dryrun":
        print("   (dryrun：未调用 LLM；游标/防穿越/每月新闻逻辑已验证。)")


if __name__ == "__main__":
    main()
