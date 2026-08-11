"""Debug why batch B yields 0 IC dates inside the full script context."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns)

panels = load_panels(days=3000)
closes = close_panel(panels)
rets = closes.pct_change()
H_ADM = 10
fwd = forward_returns(closes, H_ADM)
print("closes index range:", closes.index[0].date(), "->", closes.index[-1].date(), "shape", closes.shape)
print("fwd nonnull:", int(fwd.notna().sum().sum()))


def per_asset(func):
    out = {}
    for a in closes.columns:
        s = closes[a].dropna()
        out[a] = func(s)
    return pd.DataFrame(out).reindex(closes.index)


cands = {}
cands["reversal_5d"] = -per_asset(lambda s: s.pct_change().rolling(5).sum())
cands["skew_60d"] = per_asset(lambda s: s.pct_change().rolling(60).skew())

# ---- vectorized rank IC (exact copy from batch B) ----
def rank_ic_series_vec(factor_panel, fwd, min_valid=8):
    fr = factor_panel.rank(axis=1, method="average")
    rr = fwd.rank(axis=1, method="average")
    count = (fr.notna() & rr.notna()).astype(float)
    n = count.sum(axis=1)
    fm = fr.fillna(0.0) - (fr.fillna(0.0) * count).sum(axis=1) / n.replace(0, np.nan)
    rm = rr.fillna(0.0) - (rr.fillna(0.0) * count).sum(axis=1) / n.replace(0, np.nan)
    fm = fm.where(count > 0)
    rm = rm.where(count > 0)
    num = (fm * rm).sum(axis=1)
    den = np.sqrt((fm ** 2).sum(axis=1) * (rm ** 2).sum(axis=1))
    ic = (num / den.replace(0, np.nan)).where((n >= min_valid) & (den > 1e-14))
    return ic.dropna().rename("ic")

for name, panel in cands.items():
    panel = panel.reindex(closes.index)
    print(name, "nonnull:", int(panel.notna().sum().sum()), "ge8(factor):", int((panel.notna().sum(axis=1) >= 8).sum()))
    ics = rank_ic_series_vec(panel, fwd, 8)
    print(name, "ics len:", len(ics))
    if len(ics) > 0:
        print("  first/last ic date:", ics.index[0].date(), ics.index[-1].date(), "mean ic:", round(float(ics.mean()), 4))
