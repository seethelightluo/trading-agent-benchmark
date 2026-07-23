"""数据适配器：规范长表 → AlphaCrafter 沙箱 + FactorMiner 长表 panel。

输入契约（一张长表，CSV/Parquet 均可），列：
    date(或 datetime/trade_date), asset_id(或 symbol/ticker),
    open, high, low, close, volume [, amount]
覆盖 2020-01-01 → 2035-12-31（各WL末阶段）的全部交易日（2020~2026.7.16 为 warm-up 历史，
2026.7.16~2030 为前向世界线未来）。每资产可不等长（停牌/休市）。

输出：
  AC  沙箱  <ac_sandbox>/persistent/
        index_data/<symbol>.csv   date,open,close,high,low,volume,change,pct_change
        date.json                 {current_date: 2026-07-16, trading_days: [...]}
        account.json              初始现金 + watch_list=19 资产
        stock_news/<symbol>.json  每月 1 日宏观新闻（可选外部 monthly_news.json 注入）
  FM  工作区 <fm_workspace>/
        panel.parquet             datetime,asset_id,open,high,low,close,volume,amount

防穿越说明：AC 的数据工具天然按 date.json 的 current_date 过滤（get_stock_data /
get_index_data / get_news 只返回 ≤ current_date 的数据）；FM 由 walk_forward 在
每日 t 只切片 panel[:t]。故本适配器一次性落盘全量数据是安全的。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from asset_universe import AC_SYMBOLS, BASELINE_DATE, BY_ID, IDS
from friction import COST_BPS

# ---------------------------------------------------------------------------
# 列名归一化（兼容常见数据商导出的别名，与 FM data/loader.py 别名保持一致）
# ---------------------------------------------------------------------------
CANONICAL = {
    "date": ["datetime", "timestamp", "date", "time", "trade_date"],
    "asset_id": ["asset_id", "symbol", "ticker", "code", "stock_code", "ts_code", "instrument"],
    "open": ["open", "open_price"],
    "high": ["high", "high_price"],
    "low": ["low", "low_price"],
    "close": ["close", "close_price", "price"],
    "volume": ["volume", "vol"],
    "amount": ["amount", "amt", "turnover", "value", "traded_amount"],
}


def _rename(df: pd.DataFrame) -> pd.DataFrame:
    """把任意别名列名归一化为 canonical 名。"""
    lower = {str(c).lower().strip(): c for c in df.columns}
    rename = {}
    for canon, aliases in CANONICAL.items():
        for a in aliases:
            if a in lower and canon not in df.columns:
                rename[lower[a]] = canon
                break
    return df.rename(columns=rename)


def load_panel(path: Path) -> pd.DataFrame:
    """读取并校验规范长表，返回排序后的 DataFrame。"""
    if path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df = _rename(df)

    required = ["date", "asset_id", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"输入长表缺少列: {missing}；现有列: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    if "amount" not in df.columns:
        df["amount"] = df["close"].astype(float) * df["volume"].astype(float)
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    unknown = sorted(set(df["asset_id"]) - set(IDS))
    if unknown:
        print(f"⚠️  发现 {len(unknown)} 个不在 19 资产注册表里的 asset_id，将被忽略: {unknown[:10]}{' ...' if len(unknown) > 10 else ''}")
        df = df[df["asset_id"].isin(IDS)]
    df = df.sort_values(["asset_id", "date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# AlphaCrafter 沙箱写入
# ---------------------------------------------------------------------------
def _per_asset_index_csv(g: pd.DataFrame) -> pd.DataFrame:
    """把单资产长表转成 AC index_data 的 CSV schema。"""
    g = g.sort_values("date").copy()
    g["change"] = g["close"].diff().fillna(0.0).round(4)
    g["pct_change"] = (g["change"] / g["close"].shift(1)).fillna(0.0).round(4)
    return g[["date", "open", "close", "high", "low", "volume", "change", "pct_change"]]


def write_alpha_crafter(df: pd.DataFrame, sandbox: Path, initial_cash: float) -> None:
    persistent = sandbox / "persistent"
    # stock_data：交易所撮合（StepTool/Exchange）只从此处取价成交 → 必须写入
    # index_data：GetIndexDataTool 只读参考（Screener/Miner 用）
    (persistent / "stock_data").mkdir(parents=True, exist_ok=True)
    (persistent / "index_data").mkdir(parents=True, exist_ok=True)
    (persistent / "stock_news").mkdir(parents=True, exist_ok=True)

    written = []
    for symbol, g in df.groupby("asset_id"):
        idx = _per_asset_index_csv(g)                 # date,open,close,high,low,volume,change,pct_change
        idx.to_csv(persistent / "index_data" / f"{symbol}.csv", index=False)
        # stock_data 在 index schema 基础上补 PE/PS/PB/DYR 空列，贴合 template schema
        stk = idx.copy()
        for col in ["PE", "PS", "PB", "DYR"]:
            stk[col] = ""                              # 跨类资产无基本面
        stk.to_csv(persistent / "stock_data" / f"{symbol}.csv", index=False)
        written.append(symbol)

    # 缺数据的资产写基线占位行，避免 get_stock_data / get_index_data 报错
    for sym in AC_SYMBOLS:
        if sym not in written:
            row = {"date": BASELINE_DATE, "open": BY_ID[sym].baseline, "close": BY_ID[sym].baseline,
                   "high": BY_ID[sym].baseline, "low": BY_ID[sym].baseline, "volume": 0,
                   "change": 0.0, "pct_change": 0.0}
            pd.DataFrame([row]).to_csv(persistent / "index_data" / f"{sym}.csv", index=False)
            stk_row = {**row, "PE": "", "PS": "", "PB": "", "DYR": ""}
            pd.DataFrame([stk_row]).to_csv(persistent / "stock_data" / f"{sym}.csv", index=False)
            print(f"⚠️  资产 {sym} 无数据，已写基线占位行（stock_data+index_data）。")

    # 2) date.json：trading_days = 全 universe 日期并集；current_date = 基线日
    trading_days = sorted(df["date"].unique().tolist())
    if BASELINE_DATE not in trading_days:
        trading_days.append(BASELINE_DATE)
        trading_days = sorted(trading_days)
    with open(persistent / "date.json", "w", encoding="utf-8") as f:
        json.dump({"current_date": BASELINE_DATE, "trading_days": trading_days}, f, ensure_ascii=False)

    # 3) account.json：watch_list = 19 资产，初始现金，空仓
    account = {
        "total_assets": initial_cash,
        "net_assets": initial_cash,
        "available_cash": initial_cash,
        "market_value": 0,
        "total_profit_loss": 0,
        "total_profit_loss_rate": 0.0,
        "gross_position_rate": 0.0,
        "net_position_rate": 0.0,
        "positions": [],
        "orders": [],
        "watch_list": AC_SYMBOLS,
    }
    with open(persistent / "account.json", "w", encoding="utf-8") as f:
        json.dump(account, f, ensure_ascii=False, indent=2)

    print(f"✅ AlphaCrafter 沙箱写入 {sandbox}（{len(written)} 资产，{len(trading_days)} 交易日）")


# ---------------------------------------------------------------------------
# 每月 1 日新闻注入（AlphaCrafter Screener 在每月首个交易日读取）
# ---------------------------------------------------------------------------
def _month_keys(start: str, end: str) -> List[str]:
    """返回 [start..end] 之间所有 YYYY-MM（闭区间）。"""
    s = pd.Timestamp(start); e = pd.Timestamp(end)
    keys = []
    cur = s.to_period("M")
    while cur <= e.to_period("M"):
        keys.append(str(cur))
        cur += 1
    return keys


def write_monthly_news(df: pd.DataFrame, sandbox: Path, news_json: Optional[Path]) -> None:
    """为每个资产写 stock_news/<symbol>.json，条目日期落在每月 1 日。

    news_json（可选）schema：{ "YYYY-MM": [ {title, summary, category, sentiment}, ... ] }
    缺省时生成中性宏观占位条目，保证 Screener 每月有新闻可读、机制可跑通。
    """
    persistent = sandbox / "persistent"
    external: Dict[str, List[dict]] = {}
    if news_json and Path(news_json).exists():
        external = json.loads(Path(news_json).read_text(encoding="utf-8"))
        print(f"ℹ️  读取外部月度新闻 {news_json}（{len(external)} 个月份）")

    months = _month_keys("2026-08", "2030-12")  # 前向阶段才开始有「未来」新闻
    for sym in AC_SYMBOLS:
        asset = BY_ID[sym]
        items = []
        for m in months:
            bullets = external.get(m, [{
                "title": f"{m} 宏观环境更新",
                "summary": f"{asset.name_zh}({sym}) 所属 {asset.asset_class} 板块按当月世界线情景演进；Screener 结合常识研判 Regime。",
                "category": "Macro",
                "sentiment": "neutral",
            }])
            for b in bullets:
                items.append({
                    "publish_date": f"{m}-01 09:00:00",
                    "title": b.get("title", f"{m} 更新"),
                    "summary": b.get("summary", ""),
                    "source": b.get("source", "WorldLine"),
                    "category": b.get("category", "Macro"),
                    "sentiment": b.get("sentiment", "neutral"),
                })
        with open(persistent / "stock_news" / f"{sym}.json", "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"✅ 写入 {len(AC_SYMBOLS)} 个资产的月度新闻（{len(months)} 个月，每月 1 日）")


# ---------------------------------------------------------------------------
# FactorMiner 长表 panel
# ---------------------------------------------------------------------------
def write_factor_miner(df: pd.DataFrame, fm_workspace: Path) -> None:
    fm_workspace.mkdir(parents=True, exist_ok=True)
    panel = df.rename(columns={"date": "datetime"})[
        ["datetime", "asset_id", "open", "high", "low", "close", "volume", "amount"]
    ].copy()
    # CSV 为主（FM data/loader.py 原生支持 csv，且不依赖 pyarrow）
    panel.to_csv(fm_workspace / "panel.csv", index=False)
    try:
        panel.to_parquet(fm_workspace / "panel.parquet", index=False)
        fmt = "csv+parquet"
    except Exception as e:
        print(f"ℹ️  跳过 parquet（{type(e).__name__}），仅写 csv（FM loader 同样支持）。")
        fmt = "csv"
    print(f"✅ FactorMiner panel 写入 {fm_workspace}/panel.{fmt}"
          f"（{len(panel)} 行，{panel['asset_id'].nunique()} 资产，"
          f"{panel['datetime'].min()} ~ {panel['datetime'].max()}）")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="19 资产日频数据 → AC 沙箱 + FM panel")
    p.add_argument("--panel", required=True, help="规范长表路径（csv/parquet）")
    p.add_argument("--ac-sandbox", default="ac_sandbox_wf", help="AC 沙箱输出目录")
    p.add_argument("--fm-workspace", default="fm_workspace", help="FM 工作区输出目录")
    p.add_argument("--news", default=None, help="可选：外部月度新闻 JSON（{YYYY-MM:[...]}）")
    p.add_argument("--cash", type=float, default=1.0e7, help="初始现金（默认 1 千万）")
    args = p.parse_args()

    df = load_panel(Path(args.panel))
    print(f"载入 {len(df)} 行，{df['asset_id'].nunique()} 资产，"
          f"{df['date'].min()} ~ {df['date'].max()}")
    print(f"统一摩擦：单边 {COST_BPS}bps（已内嵌于两框架）")

    write_alpha_crafter(df, Path(args.ac_sandbox), args.cash)
    write_monthly_news(df, Path(args.ac_sandbox), Path(args.news) if args.news else None)
    write_factor_miner(df, Path(args.fm_workspace))


if __name__ == "__main__":
    main()
