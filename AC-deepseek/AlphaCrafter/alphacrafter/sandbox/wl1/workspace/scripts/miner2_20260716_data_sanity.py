"""miner_2 data sanity: verify full factor research panel up to 2026-07-15 (warm-up cut)."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import miner1_common as mc

closes = mc.load_close()  # 15 symbols, cut at 2026-07-15
print("symbols:", mc.SYMBOLS)
print("macro:", mc.MACRO)

for s, df in closes.items():
    r = df["pct_change"] if "pct_change" in df else df["close"].pct_change()
    print(f"{s:10s} rows={len(df):4d} {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d} "
          f"close_nan={df['close'].isna().sum():3d} vol_nan={(df.get('volume', pd.Series(dtype=float)).isna()).sum():3d}")

# alignment of trading calendars
common_idx = None
for s, df in closes.items():
    idx = set(df.index)
    common_idx = idx if common_idx is None else common_idx.intersection(idx)
print("\ncommon trading dates across 15 assets:", len(common_idx), "first:", min(common_idx), "last:", max(common_idx))

# macro data availability (observation-only)
print("\n--- macro (observation-only) ---")
for m in mc.MACRO:
    d = pd.read_csv(f"{mc.IDX_DIR}/{m}.csv")
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] <= mc.CUT]
    print(f"{m:7s} rows={len(d):4d} {d['date'].min():%Y-%m-%d} -> {d['date'].max():%Y-%m-%d}")

# volume sanity: are volumes meaningful across assets?
print("\n--- daily volume/market cap scale (median last 60d) ---")
for s, df in closes.items():
    v = df["volume"].tail(60) if "volume" in df else None
    if v is not None and v.notna().sum() > 0:
        print(f"{s:10s} med_volume={v.median():.3e}")
    else:
        print(f"{s:10s} volume: n/a")