#!/usr/bin/env python
"""
make_panel.py — 从 asset-daily-data/<asset_id>.csv 重建规范长表 panel.parquet/csv

无需重新联网抓取；当只改了面板口径（如资产子集、warmup 截止日）时用这个。
fetch_daily_data.py 末尾已内联同样的逻辑，本脚本是"只重组件"的轻量入口。

用法
  python make_panel.py                    # 19 基准资产，≤2026-07-16
  python make_panel.py --all              # 含 KOSPI/USDKRW/JP_SEMI_EQUIP 等全部
  python make_panel.py --end 2026-07-16
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "asset-daily-data"

from asset_spec import ASSET_SPEC, BENCHMARK_ASSET_IDS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default=str(OUTDIR))
    ap.add_argument("--end", default="2026-07-16", help="面板 warmup 截止日")
    ap.add_argument("--all", action="store_true", help="包含全部资产(含WL特有)")
    args = ap.parse_args()

    indir = Path(args.indir)
    keep = [s["asset_id"] for s in ASSET_SPEC] if args.all else BENCHMARK_ASSET_IDS

    parts, missing = [], []
    for aid in keep:
        p = indir / f"{aid}.csv"
        if not p.exists():
            missing.append(aid)
            continue
        df = pd.read_csv(p)
        df.insert(0, "asset_id", aid)
        if "adjclose" in df.columns:
            df = df.drop(columns=["adjclose"])
        df["amount"] = (pd.to_numeric(df["close"], errors="coerce")
                        * pd.to_numeric(df["volume"], errors="coerce").fillna(0)).fillna(0)
        for c in ("open", "high", "low", "close", "volume", "amount"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        parts.append(df)

    if not parts:
        raise SystemExit(f"未找到任何 CSV (indir={indir})")
    panel = (pd.concat(parts, ignore_index=True)
             .sort_values(["asset_id", "date"]).reset_index(drop=True))
    panel = panel[panel["date"] <= args.end].reset_index(drop=True)

    pq = indir / "panel.parquet"
    try:
        panel.to_parquet(pq, index=False)
        print(f"✅ panel.parquet  ({len(panel)} 行, {panel['asset_id'].nunique()} 资产, ≤{args.end})")
    except Exception as e:
        print(f"⚠️  parquet 写入失败 ({type(e).__name__}): {e}")
    panel.to_csv(indir / "panel.csv", index=False)
    print(f"✅ panel.csv      ({len(panel)} 行)")
    if missing:
        print(f"⚠️  缺 CSV: {missing}")


if __name__ == "__main__":
    main()
