"""
build_inputs.py — 20 资产（15 可交易 + 5 信号）日频数据 → 两套框架的输入

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
   persistent/date.json                   {current_date: 执行日, visible_through: 可见行情截止日, ...}
   persistent/account.json                初始现金 + watch_list=15 可交易 + 空持仓
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
                        session_id: str, template: str = "template_a",
                        stage_news: list | None = None) -> Path:
    repo = HERE / "AlphaCrafter" / "alphacrafter" / "sandbox"
    src = repo / template
    dst = repo / session_id
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    persistent = dst / "persistent"
    workspace = dst / "workspace"
    stock_data = persistent / "stock_data"      # 可交易持仓（tradable，15）
    index_data = persistent / "index_data"      # 宏观/状态信号（signals，5；Screener 参考，不持仓）
    news_dir = persistent / "stock_news"
    for d in (stock_data, index_data, news_dir):
        d.mkdir(exist_ok=True)
    for d in (stock_data, index_data):
        for f in d.glob("*.csv"):
            f.unlink()
    for f in news_dir.glob("*.json"):
        f.unlink()

    # Miners run concurrently and may write their first factor/script at the same
    # time.  Create the shared targets before any agent starts instead of making
    # each agent race to create them through the shell tool.
    for d in (workspace / "factors", workspace / "scripts"):
        d.mkdir(parents=True, exist_ok=True)

    tradable_ids = [a["asset_id"] for a in assets_cfg["tradable"]]
    signal_ids = [a["asset_id"] for a in assets_cfg.get("signals", [])]

    def _write_csv(aid, out_dir, with_fund):
        sub = panel[panel["asset_id"] == aid].copy().sort_values("date")
        if sub.empty:
            print(f"  ⚠️  资产 {aid} 在面板中无数据，跳过")
            return
        sub["change"] = sub["close"].diff()
        sub["pct_change"] = sub["close"].pct_change()
        out = sub[["date", "open", "close", "high", "low", "volume",
                   "change", "pct_change"]].copy()
        if with_fund:  # stock_data 含空 PE/PS/PB/DYR 列；index_data 不含
            out["PE"] = out["PS"] = out["PB"] = out["DYR"] = np.nan
            out = out[["date", "open", "close", "high", "low", "volume",
                       "change", "pct_change", "PE", "PS", "PB", "DYR"]]
        out.to_csv(out_dir / f"{aid}.csv", index=False)

    # 1) stock_data：可交易持仓（15，含空 fundamentals）
    for aid in tradable_ids:
        _write_csv(aid, stock_data, with_fund=True)
    # 1b) index_data：宏观/状态信号（5，不持仓，供 Screener GetIndexDataTool 读取）
    for aid in signal_ids:
        _write_csv(aid, index_data, with_fund=False)

    # 2) date.json：warm-up 只研究、不交易。2026-07-16 为首个执行日，但日线数据
    #    只能看到前一交易日（通常为 2026-07-15），避免用当天收盘价做当天交易。
    trading_days = sorted(panel["date"].unique().tolist())
    baseline = assets_cfg["baseline_date"]
    if baseline not in trading_days:
        # 取基准日当天或之后的第一个交易日
        baseline = next((d for d in trading_days if d >= baseline), trading_days[0])
    baseline_idx = trading_days.index(baseline)
    if baseline_idx == 0:
        raise ValueError(
            f"panel must contain at least one warm-up trading day before {baseline}"
        )
    visible_through = trading_days[baseline_idx - 1]
    configured_history_end = assets_cfg.get("history_end")
    if configured_history_end and configured_history_end != visible_through:
        raise ValueError(
            "history_end must equal the last panel trading day before the first "
            f"execution day: configured={configured_history_end}, actual={visible_through}"
        )
    date_json = {
        "current_date": baseline,
        "visible_through": visible_through,
        "simulation_complete": False,
        "trading_days": trading_days,
    }
    (persistent / "date.json").write_text(
        json.dumps(date_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) account.json：初始现金 + watch_list = 15 可交易持仓 + 空持仓/订单
    initial_capital = float(assets_cfg.get("initial_capital_usd", 100_000_000.0))
    account = {
        "initial_capital": initial_capital,
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
        "watch_list": tradable_ids,   # 仅可交易持仓（15）；信号不进 watch_list
    }
    (persistent / "account.json").write_text(
        json.dumps(account, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4) 新闻注入。优先用世界线 stage_news（对齐阶段事件，dated 在 news_date=价格 leak 之后）；
    #    无则回退每月 1 日占位新闻。AC get_news 按 publish_date ≤ current_date 过滤。
    if stage_news:
        def _items_for(aid):
            out = []
            for st in stage_news:
                endpts = st.get("asset_endpoints", {}) or {}
                if aid not in endpts:
                    continue
                tgt = endpts[aid]
                out.append({
                    "publish_date": f"{st['news_date']} 09:00:00",   # news_date 滞后价格 leak
                    "title": f"[{aid}] {st.get('title','世界线阶段事件')}",
                    "summary": (f"世界线阶段「{st.get('title','')}」触发（阶段末 {st.get('stage_end')}）。"
                                f"该资产阶段末目标 {tgt}。注：价格已提前反映（内幕 leak），本新闻为公开确认。"),
                    "source": "worldline-stage", "category": "Macro",
                    "sentiment": "neutral",
                })
            return out
        news_src = "stage_news（对齐世界线，滞后 leak）"
    else:
        months = _month_first_days(baseline, assets_cfg["online_end"])
        def _items_for(aid):
            return [{
                "publish_date": f"{m} 09:00:00",
                "title": f"[{aid}] monthly macro brief ({m[:7]}) — 待填世界线叙事",
                "summary": ("占位新闻：每月 1 日注入。请用对应世界线该月的宏观因果替换本条。"),
                "source": "worldline-injection", "category": "Macro", "sentiment": "neutral",
            } for m in months]
        news_src = f"monthly 占位（{len(months)} 月）"
    n_news = 0
    for aid in tradable_ids:   # 新闻仅注入可交易持仓
        items = _items_for(aid)
        n_news = max(n_news, len(items))
        (news_dir / f"{aid}.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  ✅ AC session: {dst}")
    print(f"     trading_days={len(trading_days)}  current_date={baseline}  "
          f"visible_through={visible_through}  initial_capital=${initial_capital:,.0f}  "
          f"tradable={len(tradable_ids)}(stock_data)  signals={len(signal_ids)}(index_data)  "
          f"news={news_src} (~{n_news}/资产)")
    return dst


# --------------------------------------------------------------------------- #
# FactorMiner
# --------------------------------------------------------------------------- #
def build_factor_miner(panel: pd.DataFrame, assets_cfg: dict,
                       fm_data_dir: Path) -> Path:
    fm_data_dir.mkdir(parents=True, exist_ok=True)
    # FM 的横截面资产只能是可交易宇宙。观察型信号（VIX、利率、FX 等）若混在
    # asset_id 维度中，会被错误地当成候选持仓并污染 IC/分位数组合。
    tradable_ids = {str(item["asset_id"]) for item in assets_cfg["tradable"]}
    fm = panel[panel["asset_id"].astype(str).isin(tradable_ids)].copy()
    if fm.empty:
        raise ValueError("FactorMiner panel has no rows from the tradable universe")

    # FM loader 期望长表：datetime, asset_id, open, high, low, close, volume, amount
    fm = fm.rename(columns={"date": "datetime"})
    fm["datetime"] = pd.to_datetime(fm["datetime"])
    keep = ["datetime", "asset_id", "open", "high", "low", "close", "volume", "amount"]
    fm = fm[keep].sort_values(["asset_id", "datetime"]).reset_index(drop=True)
    panel_path = fm_data_dir / "panel.parquet"
    fm.to_parquet(panel_path, index=False)

    # 生成一份 walkforward 配置，继承 default.yaml 并指向本面板
    cfg_src = HERE / "FactorMiner" / "factorminer" / "configs" / "default.yaml"
    cfg_dst = HERE / "FactorMiner" / "factorminer" / "configs" / "walkforward.yaml"
    text = cfg_src.read_text(encoding="utf-8")
    note = ("# [walk-forward] 前向走步配置；execution.cost_bps=3.0 已统一为单边 3bps。\n"
            f"# 面板：data/panel.parquet（{len(fm)} 行，{fm['asset_id'].nunique()} 资产，"
            "仅含可交易资产；观察型信号不伪装成横截面资产）\n")
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
    ap.add_argument("--stage-news", default="",
                    help="WL<n>_stage_news.json 路径；提供则按世界线阶段事件注入对齐 news（滞后价格 leak）")
    args = ap.parse_args()

    assets_cfg = load_assets(Path(args.assets))
    panel = load_panel(Path(args.panel))
    print(f"载入面板：{len(panel)} 行，{panel['asset_id'].nunique()} 资产，"
          f"{panel['date'].min()} ~ {panel['date'].max()}")

    stage_news = None
    if args.stage_news:
        import json as _json
        stage_news = _json.loads(Path(args.stage_news).read_text(encoding="utf-8"))
        print(f"载入阶段新闻：{len(stage_news)} 条（{stage_news[0].get('news_date') if stage_news else '-'} 起）")

    if not args.skip_ac:
        build_alpha_crafter(panel, assets_cfg, args.ac_session, args.ac_template,
                            stage_news=stage_news)
    if not args.skip_fm:
        build_factor_miner(panel, assets_cfg, Path(args.fm_dir))


if __name__ == "__main__":
    main()
