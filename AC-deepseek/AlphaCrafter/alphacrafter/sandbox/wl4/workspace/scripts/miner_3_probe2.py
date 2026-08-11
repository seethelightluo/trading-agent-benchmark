import sys, time
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import load_panels, close_panel, forward_returns, rank_ic_series

t0 = time.time()
panels = load_panels(days=3000)
closes = close_panel(panels)
rets = closes.pct_change()
print(f"closes shape {closes.shape}, NaNs per col:\n{closes.isna().sum()}", flush=True)
print("last dates per col:")
print(closes.apply(lambda s: s.dropna().index.max()).to_string(), flush=True)
print("first dates per col:")
print(closes.apply(lambda s: s.dropna().index.min()).to_string(), flush=True)

fwd = forward_returns(closes, 10)
sig = rets.rolling(20).std()
valid = sig.notna() & fwd.notna()
nvalid = valid.sum(axis=1)
print(f"\nvalid-count distribution: min={nvalid.min()} p10={nvalid.quantile(0.1)} med={nvalid.median()} max={nvalid.max()}")
print(f"dates with >=8 valid: {(nvalid >= 8).sum()} / {len(nvalid)}")

# fast vectorized rank IC
def rank_ic_fast(factor_panel, fwd, min_valid=8):
    fr = factor_panel.rank(axis=1)
    rr = fwd.rank(axis=1)
    valid = factor_panel.notna() & fwd.notna()
    n = valid.sum(axis=1)
    frc = fr.sub(fr.mean(axis=1), axis=0)
    rrc = rr.sub(rr.mean(axis=1), axis=0)
    num = (frc * rrc).sum(axis=1)
    den = np.sqrt((frc ** 2).sum(axis=1) * (rrc ** 2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    ic = ic.where(n >= min_valid)
    return ic.dropna()

t1 = time.time()
ics_slow = rank_ic_series(sig, fwd, 8)
print(f"\nslow rank_ic: {time.time()-t1:.1f}s n={len(ics_slow)} mean={ics_slow.mean():.4f}")
t1 = time.time()
ics_fast = rank_ic_fast(sig, fwd, 8)
print(f"fast rank_ic: {time.time()-t1:.3f}s n={len(ics_fast)} mean={ics_fast.mean():.4f}")

common = ics_slow.index.intersection(ics_fast.index)
diff = (ics_slow[common] - ics_fast[common]).abs()
print(f"common {len(common)} dates; max abs diff {diff.max():.6f} mean {diff.mean():.6f}")
