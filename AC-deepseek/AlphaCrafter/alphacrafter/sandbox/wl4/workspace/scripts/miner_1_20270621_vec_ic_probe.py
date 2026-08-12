"""miner_1 data coverage check + vectorized rank_ic correctness probe (2027-06-21)."""
import sys, time
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, TRADABLE)

t0 = time.time()
panels = load_panels(days=3000)
closes = close_panel(panels)
rets = closes.pct_change()
print(f"closes {closes.shape} {closes.index.min().date()}..{closes.index.max().date()}", flush=True)

# --- NaN coverage per asset ---
nan_cnt = closes.isna().sum()
print("\nNaN per asset:\n", nan_cnt[nan_cnt > 0].to_string() if (nan_cnt > 0).any() else "none")

# frozen check
for a in ["HSI", "ETH"]:
    s = closes[a].dropna()
    n_flat = int((s.diff() == 0).sum())
    print(f"{a}: n={len(s)} flat_days={n_flat} last={s.index.max().date()} last_close={s.iloc[-1]:.4f}")

# --- vectorized rank IC ---
def rank_ic_vec(factor_panel, fwd, min_valid=8):
    """Row-wise Spearman IC via cross-sectional ranks (fully vectorized)."""
    f = factor_panel.rank(axis=1, method="average")
    r = fwd.reindex(factor_panel.index).rank(axis=1, method="average")
    valid = factor_panel.notna() & fwd.reindex(factor_panel.index).notna()
    nv = valid.sum(axis=1)
    # mean-center + scale over valid entries only
    fz = f.sub(f.mean(axis=1), axis=0).div(f.std(axis=1, ddof=1).replace(0, np.nan), axis=0)
    rz = r.sub(r.mean(axis=1), axis=0).div(r.std(axis=1, ddof=1).replace(0, np.nan), axis=0)
    fz = fz.where(valid).fillna(0.0)
    rz = rz.where(valid).fillna(0.0)
    # Pearson corr per row = dot(z_f, z_r) / (nv-1)
    denom = nv - 1
    ic = (fz * rz).sum(axis=1) / denom
    ic = ic.where((nv >= min_valid) & (denom > 0))
    ic = ic.dropna()
    ic.name = "ic"
    return ic

fwd10 = forward_returns(closes, 10)
fac = rets.rolling(20).std()

t1 = time.time()
ics_slow = rank_ic_series(fac, fwd10, 8)
print(f"\nslow rank_ic: {time.time()-t1:.1f}s n={len(ics_slow)} mean={ics_slow.mean():.4f}")
t1 = time.time()
ics_fast = rank_ic_vec(fac, fwd10, 8)
print(f"fast rank_ic: {time.time()-t1:.1f}s n={len(ics_fast)} mean={ics_fast.mean():.4f}")

if len(ics_slow) and len(ics_fast):
    joined = pd.concat([ics_slow.rename("slow"), ics_fast.rename("fast")], axis=1).dropna()
    diff = (joined["slow"] - joined["fast"]).abs()
    print(f"match: n_common={len(joined)} max_abs_diff={diff.max():.6f} mean_abs_diff={diff.mean():.6f}")
    # where do they diverge in count?
    print("slow-only dates:", len(set(ics_slow.index) - set(ics_fast.index)),
          "fast-only dates:", len(set(ics_fast.index) - set(ics_slow.index)))

# n_ic_dates for a standard library factor
mom = closes.shift(5) / closes.shift(15) - 1.0
ics_m = rank_ic_vec(mom, fwd10, 8)
print(f"\nmom_10d_skip5 n_ic={len(ics_m)} mean={ics_m.mean():.4f} icir={ics_m.mean()/ics_m.std(ddof=1):.4f}")
