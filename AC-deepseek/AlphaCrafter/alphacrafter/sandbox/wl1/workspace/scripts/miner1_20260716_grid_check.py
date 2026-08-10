"""miner_1 2026-07-16: verify canonical signal-artifact grid (dates x symbols)."""
import sys, os, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close, CUT

t0 = time.time()
closes = load_close()
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[(idx >= pd.Timestamp("2020-01-01")) & (idx <= CUT)]
print(f"common dates 2020+: {len(idx)}  {idx.min().date()} .. {idx.max().date()}")

idx21 = idx[idx >= pd.Timestamp("2021-01-01")]
print(f"common dates 2021+: {len(idx21)}  {idx21.min().date()} .. {idx21.max().date()}")

# per-symbol coverage on idx21
full = pd.DataFrame({s: closes[s]['close'].reindex(idx21).astype(float) for s in SYMBOLS})
print("NaN cells on idx21 grid:", int(full.isna().sum().sum()), "of", full.size)
print("columns order:", list(full.columns))
print(f"time {time.time()-t0:.1f}s")
