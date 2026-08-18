"""miner_1 focused validation: ret_autocorr_60d (return persistence factor).

Hypothesis: assets whose daily returns are positively autocorrelated (trending
microstructure) vs negatively autocorrelated (mean-reverting) have different
forward return profiles. Orthogonal to level-based momentum (rho~0.1 vs lib).

Validation: full-sample IC/ICIR by horizon, decay, regime splits, turnover,
coverage, library correlation provenance.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (load_close, load_macro, daily_ic, ic_stats,
                          rank_turnover, coverage_stats, library_panel,
                          max_lib_corr)

END = "2028-10-23"
close = load_close(END)
macro = load_macro(END)
ret = close.pct_change()
fwd10 = close.shift(-10) / close - 1.0


def autocorr_60(x):
    s = pd.Series(x)
    return s.autocorr() if len(x) > 3 else np.nan


F = ret.rolling(60, min_periods=30).apply(autocorr_60, raw=False)
F.name = "ret_autocorr_60d"
print(f"universe={close.shape[1]} assets, dates={len(close)} "
      f"({close.index[0].date()} -> {close.index[-1].date()})")

print("\n=== Horizon decay (full sample) ===")
hor = {}
for h in (1, 2, 3, 5, 10, 20):
    fwd = close.shift(-h) / close - 1.0
    ic = daily_ic(F, fwd)
    st = ic_stats(ic, h)
    hor[h] = st
    print(f"h={h:>2d}: IC={st['ic']:+.4f}  ICIR={st['icir']:+.4f}  "
          f"hit={st['hit']:.3f}  n={st['n']}")

print("\n=== Regime splits (h=10) ===")
splits = [("2020-2021", "2020-01-01", "2021-12-31"),
          ("2022-2023", "2022-01-01", "2023-12-31"),
          ("2024-2025", "2024-01-01", "2025-12-31"),
          ("2026-2028H1", "2026-01-01", "2028-06-30"),
          ("recent 1y", "2027-10-24", "2028-10-23"),
          ("recent 6m", "2028-04-24", "2028-10-23")]
for name, a, b in splits:
    sl = (F.index >= a) & (F.index <= b)
    if sl.sum() == 0:
        continue
    ic = daily_ic(F.loc[sl], fwd10.loc[sl])
    st = ic_stats(ic, 10)
    print(f"{name:>12s}: IC={st['ic']:+.4f}  ICIR={st['icir']:+.4f}  "
          f"hit={st['hit']:.3f}  n={st['n']}")

print("\n=== IC distribution (h=10, full) ===")
ic_all = daily_ic(F, fwd10).dropna()
print(f"mean={ic_all.mean():+.4f} std={ic_all.std():.4f} "
      f"median={ic_all.median():+.4f} p5={ic_all.quantile(.05):+.4f} "
      f"p95={ic_all.quantile(.95):+.4f}")

print("\n=== Coverage / turnover ===")
cov_ = coverage_stats(F, fwd10)
print(f"coverage_asset_days={cov_['coverage_asset_days']:.3f} "
      f"coverage_dates_ge8={cov_['coverage_dates_ge8']:.3f}")
for w in (5, 10, 20):
    print(f"rank_turnover({w}d)={rank_turnover(F, window=w):.3f}")

print("\n=== Library correlation provenance ===")
lib = library_panel(close, macro)
rho, pairs = max_lib_corr(F, lib)
print(f"max_abs_library_correlation={rho:.4f}")
for k, v in sorted(pairs.items(), key=lambda kv: -abs(kv[1])):
    print(f"  {k}: {v:+.4f}")

print("\n=== Factor tail snapshot (last 3 dates) ===")
print(F.tail(3).round(3).to_string())
