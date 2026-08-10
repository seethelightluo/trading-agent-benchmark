"""miner_3 decay & final validation for passing reversal / CLV candidates - 2026-07-16.
Computes IC at multiple horizons (decay), Pearson vs rank IC, and library correlation audit."""
import sys, os, time
import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close
import miner3_fast as F

t0 = time.time()
closes = load_close()
idx = None
for s, df in closes.items():
    idx = df.index if idx is None else idx.intersection(df.index)
idx = idx[(idx >= pd.Timestamp("2020-01-01"))]
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in SYMBOLS})
RET = CP.pct_change()
print(f"loaded {len(idx)} common dates {idx.min().date()}..{idx.max().date()} ({time.time()-t0:.1f}s)")

panels = {
    "rev_1d": -RET,
    "rev_2d": -(CP / CP.shift(2) - 1.0),
    "clv_5d": (CP - CP.rolling(5).min()) / (CP.rolling(5).max() - CP.rolling(5).min() + 1e-12),
}

print("\n=== decay: IC / ICIR at horizons 1..30 (fast_ic_all) ===")
for name, p in panels.items():
    p = p.reindex(idx)
    out = F.fast_ic_all(p, closes, horizons=(1, 2, 3, 5, 10, 20, 30))
    s = "  ".join(f"h{h}:{out[h]['ic']:+.4f}/{out[h]['icir']:+.3f}" for h in out)
    print(f"{name:8s} {s}")

print("\n=== rank IC (Spearman) at h=1 and h=10 ===")
fwd1 = F.fwd_returns(closes, 1).reindex(idx)
fwd10 = F.fwd_returns(closes, 10).reindex(idx)
for name, p in panels.items():
    p = p.reindex(idx)
    ics1, ics10 = [], []
    for d in idx:
        x = p.loc[d].dropna()
        y1 = fwd1.loc[d].reindex(x.index).dropna()
        y10 = fwd10.loc[d].reindex(x.index).dropna()
        if len(x) >= 8 and len(y1) >= 8 and x.std() > 0 and y1.std() > 0:
            ics1.append(st.spearmanr(x, y1)[0])
        if len(x) >= 8 and len(y10) >= 8 and x.std() > 0 and y10.std() > 0:
            ics10.append(st.spearmanr(x, y10)[0])
    a = np.array(ics1); b = np.array(ics10)
    print(f"{name:8s} rankIC1={a.mean():+.4f} ICIR={a.mean()/a.std():+.3f} n={len(a)} | "
          f"rankIC10={b.mean():+.4f} ICIR={b.mean()/b.std():+.3f} n={len(b)}")

print("\n=== library correlation audit ===")
lib = [f for f in os.listdir("factors") if f.endswith(".json")]
print(f"library factors found: {len(lib)}")
max_abs = 0.0
if lib:
    stk = pd.DataFrame({n: p.reindex(idx).stack() for n, p in panels.items()}).dropna()
    for lf in lib:
        import json
        meta = json.load(open(f"factors/{lf}"))
        print(f"  library factor {lf}: id={meta.get('factor_id')}")
print(f"max_abs_library_correlation = {max_abs:.3f} (no library factors yet)")

print(f"\n[elapsed {time.time()-t0:.1f}s]")
