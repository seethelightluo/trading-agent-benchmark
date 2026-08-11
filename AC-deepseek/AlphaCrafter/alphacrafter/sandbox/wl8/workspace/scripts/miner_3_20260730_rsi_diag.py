"""miner_3 2026-07-30 — Investigate RSI panel artifacts and library correlation."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_3_20260730_common import (
    get_watchlist, load_data, max_library_corr, library_factors,
)

data = load_data(days=3200)


def rsi_14(c, win=14):
    d = c.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    au = up.ewm(alpha=1.0 / win, adjust=False).mean()
    ad = dn.ewm(alpha=1.0 / win, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


panel = {}
for a, d in data.items():
    panel[a] = rsi_14(d["close"].astype(float))

fdf = pd.DataFrame(panel)
print("panel shape:", fdf.shape)
print("any inf:", bool(np.isinf(fdf.values).any()))
print("any nan:", int(np.isnan(fdf.values).sum()))
print("value stats: min=%.3f max=%.3f mean=%.3f" % (np.nanmin(fdf.values), np.nanmax(fdf.values), np.nanmean(fdf.values)))
# per-asset count of non-finite
nf = fdf.notna().sum()
print("non-null per asset:\n", nf)
bad = fdf.isna().sum()
print("nan per asset (top):\n", bad[bad > 0].sort_values(ascending=False).head(10))

lib = library_factors(data)
maxrho, rho_map = max_library_corr(panel, data)
print("library corr:", {k: round(v, 4) for k, v in rho_map.items()}, "max_abs=", maxrho)

# also compute rank-based (spearman) library correlation as robustness
from scipy.stats import spearmanr
fstack = fdf.stack()
fstack = fstack[fstack.notna()]
for fid, lf in lib.items():
    ldf = pd.DataFrame(lf).stack()
    both = fstack.index.intersection(ldf.index)
    rho, _ = spearmanr(fstack.loc[both].values, ldf.loc[both].values)
    print(f"  spearman rho vs {fid}: {rho:.4f} (n={len(both)})")
