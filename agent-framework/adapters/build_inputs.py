"""
build_inputs.py — 19 资产日频数据 → 两套框架的输入

把一张"规范长表"一次性适配成 AlphaCrafter 的 session 目录 与 FactorMiner 的面板，
使两套框架都能接受 2020-2025（及 2026-2030）的日频资产数据。

规范长表契约（输入）
-------------------
CSV 或 Parquet，长格式，每行一个 (日期, 资产)：
    date, asset_id, open, high, low, close, volume[, amount]
- date:      YYYY-MM-DD
- asset_id:  与 ASSETS.yaml 中的 asset_id 一致（如 000300.SH / SPX / BTC / US10Y …）
- OHLCV:     数值；非权益资产（债券收益率/汇率/VIX）若无 volume，填 0 或 NaN 均可
- amount:    可选，成交额；缺失时用 close*volume 补

输出
----
1) AlphaCrafter session 目录（由 template_a 复制改造）：
   persistent/stock_data/<asset_id>.csv   每资产日线（AC 表头 + 空 PE/PS/PB/DYR）
   persistent/date.json                   {current_date: 基准日, trading_days: [...]}
   persistent/account.json                初始现金 + watch_list=19 资产 + 空持仓
   persistent/stock_news/<asset_id>.json  每月 1 日宏观新闻（占位，可后续填世界线叙事）
2) FactorMiner 面板：data/panel.parquet（FM loader 原生长表格式）+ walkforward.yaml 配置

用法
----
    python -m adapters.build_inputs \
        --panel ../data-prepare/data2020-2026/panel.parquet \
        --assets ASSETS.yaml \
        --ac-session wf19 \
        --fm-dir FactorMiner/data
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent.parent  # agent-framework/


def load_assets(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_panel(path: Path) -> pd.DataFrame:
    """读取规范长表，标准化列名，按 (asset_id, date) 排序。"""
    if path.suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    rename = {"date": "date", "ticker": "asset_id", "symbol": "asset_id",
              "code": "asset_id", "ts_code": "asset_id"}
    df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "amount" not in df.columns:
        df["amount"] = df["close"] * df["volume"].fillna(0)
    df = df.sort_values(["asset_id", "date"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# AlphaCrafter
# --------------------------------------------------------------------------- #
def build_alpha_crafter(panel: pd.DataFrame, assets_cfg: dict,
                        session_id: str, template: str = "template_a") -> Path:
    repo = HERE / "AlphaCrafter" / "alphacrafter" / "sandbox"
    src = repo / template
    dst = repo / session_id
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    persistent = dst / "persistent"
    stock_data = persistent / "stock_data"
    news_dir = persistent / "stock_news"
    stock_data.mkdir(exist_ok=True)
    news_dir.mkdir(exist_ok=True)
    # 清掉模板占位文件
    for f in stock_data.glob("*.csv"):
        f.unlink()
    for f in news_dir.glob("*.json"):
        f.unlink()

    asset_ids = [a["asset_id"] for a in assets_cfg["assets"]]

    # 1) 每资产日线 CSV（AC 表头；fundamentals 留空，跨资产无基本面）
    ac_header = ["date", "open", "close", "high", "low", "volume",
                 "change", "pct_change", "PE", "PS", "PB", "DYR"]
    for aid in asset_ids:
        sub = panel[panel["asset_id"] == aid].copy().sort_values("date")
        if sub.empty:
            print(f"  ⚠️  资产 {aid} 在面板中无数据，跳过")
            continue
        sub["change"] = sub["close"].diff()
        sub["pct_change"] = sub["close"].pct_change()
        out = sub[["date", "open", "close", "high", "low", "volume",
                   "change", "pct_change"]].copy()
        out["PE"] = out["PS"] = out["PB"] = out["DYR"] = np.nan
        out = out[ac_header]
        out.to_csv(stock_data / f"{aid}.csv", index=False)

    # 2) date.json：trading_days = 面板全量日期升序；current_date = 基准日（前向起点）
    trading_days = sorted(panel["date"].unique().tolist())
    baseline = assets_cfg["baseline_date"]
    if baseline not in trading_days:
        # 取基准日当天或之后的第一个交易日
        baseline = next((d for d in trading_days if d >= baseline), trading_days[0])
    date_json = {"current_date": baseline, "trading_days": trading_days}
    (persistent / "date.json").write_text(
        json.dumps(date_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) account.json：初始现金 + watch_list = 19 资产 + 空持仓/订单
    initial_capital = 10_000_000.0
    account = {
        "total_assets": initial_capital,
        "net_assets": initial_capital,
        "available_cash": initial_capital,
        "market_value": 0,
        "total_profit_loss": 0,
        "total_profit_loss_rate": 0.0,
        "gross_position_rate": 0.0,
        "net_position_rate": 0.0,
        "positions": [],
        "orders": [],
        "watch_list": asset_ids,
    }
    (persistent / "account.json").write_text(
        json.dumps(account, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4) 每月 1 日宏观新闻（占位）。AC 的 get_news 按 publish_date ≤ current_date 过滤，
    #    1 日发布当月即可被 Screener 读到 → 实现"每月 1 日注入新闻"。
    months = _month_first_days(baseline, assets_cfg["online_end"])
    for aid in asset_ids:
        items = [{
            "publish_date": f"{m} 09:00:00",
            "title": f"[{aid}] monthly macro brief ({m[:7]}) — 待填世界线叙事",
            "summary": ("占位新闻：每月 1 日注入。请用对应世界线（WL1…WL9）"
                        "该月的宏观因果与该资产的关键驱动替换本条。"),
            "source": "worldline-injection",
            "category": "Macro",
            "sentiment": "neutral",
        } for m in months]
        (news_dir / f"{aid}.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  ✅ AC session: {dst}")
    print(f"     trading_days={len(trading_days)}  current_date={baseline}  "
          f"assets={len(asset_ids)}  months_of_news={len(months)}")
    return dst


# --------------------------------------------------------------------------- #
# FactorMiner
# --------------------------------------------------------------------------- #
def build_factor_miner(panel: pd.DataFrame, assets_cfg: dict,
                       fm_data_dir: Path) -> Path:
    fm_data_dir.mkdir(parents=True, exist_ok=True)
    # FM loader 期望长表：datetime, asset_id, open, high, low, close, volume, amount
    fm = panel.rename(columns={"date": "datetime"}).copy()
    fm["datetime"] = pd.to_datetime(fm["datetime"])
    keep = ["datetime", "asset_id", "open", "high", "low", "close", "volume", "amount"]
    fm = fm[keep].sort_values(["asset_id", "datetime"]).reset_index(drop=True)
    panel_path = fm_data_dir / "panel.parquet"
    fm.to_parquet(panel_path, index=False)

    # 生成一份 walkforward 配置，继承 default.yaml 并指向本面板
    cfg_src = HERE / "FactorMiner" / "factorminer" / "configs" / "default.yaml"
    cfg_dst = HERE / "FactorMiner" / "factorminer" / "configs" / "walkforward.yaml"
    text = cfg_src.read_text(encoding="utf-8")
    note = ("# [walk-forward 小修] 19 资产前向走步配置；"
            "execution.cost_bps=3.0 已在 default.yaml 统一为单边 3bps。\n"
            f"# 面板：data/panel.parquet（{len(fm)} 行，"
            f"{fm['asset_id'].nunique()} 资产）\n")
    cfg_dst.write_text(note + text, encoding="utf-8")

    print(f"  ✅ FM panel: {panel_path}  ({len(fm)} rows, "
          f"{fm['asset_id'].nunique()} assets)")
    print(f"     FM config: {cfg_dst}")
    return panel_path


# --------------------------------------------------------------------------- #
def _month_first_days(start: str, end: str) -> list[str]:
    """从 start 所在月的下一个月 1 号，到 end，生成每月 1 号日期列表（YYYY-MM-DD）。"""
    s = pd.to_datetime(start).to_period("M").to_timestamp() + pd.offsets.MonthBegin(1)
    e = pd.to_datetime(end)
    out = []
    cur = s
    while cur <= e:
        out.append(cur.strftime("%Y-%m-%d"))
        cur = cur + pd.offsets.MonthBegin(1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", required=True, help="规范长表 parquet/csv 路径")
    ap.add_argument("--assets", default=str(HERE / "ASSETS.yaml"))
    ap.add_argument("--ac-session", default="wf19", help="AlphaCrafter session id")
    ap.add_argument("--ac-template", default="template_a",
                    choices=["template_a", "template_us"])
    ap.add_argument("--fm-dir", default=str(HERE / "FactorMiner" / "data"))
    ap.add_argument("--skip-ac", action="store_true")
    ap.add_argument("--skip-fm", action="store_true")
    args = ap.parse_args()

    assets_cfg = load_assets(Path(args.assets))
    panel = load_panel(Path(args.panel))
    print(f"载入面板：{len(panel)} 行，{panel['asset_id'].nunique()} 资产，"
          f"{panel['date'].min()} ~ {panel['date'].max()}")

    if not args.skip_ac:
        build_alpha_crafter(panel, assets_cfg, args.ac_session, args.ac_template)
    if not args.skip_fm:
        build_factor_miner(panel, assets_cfg, Path(args.fm_dir))


if __name__ == "__main__":
    main()
