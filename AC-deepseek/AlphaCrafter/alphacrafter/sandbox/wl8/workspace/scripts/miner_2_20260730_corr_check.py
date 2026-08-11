"""miner_2 2026-07-30 -- pairwise spearman correlation among new candidates and library.
Informs which candidates to persist (avoid persisting near-duplicates that will
just be evicted by the audit's pairwise correlation gate)."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, factor_panel,
                                   load_library_panels)

close, vol, open_, high, low = load_closes()
macro = {k: load_index(k) for k in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]}
for anchor in ["SPX", "XAU", "BTC", "WTI", "NDX", "US10Y"]:
    macro[anchor] = close[anchor].dropna()
lib = load_library_panels()


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / b


def f_body_ratio_10(c, v, o, h, l, m, win=10):
    rng = (h - l).replace(0, np.nan)
    return _safe_div((c - o).abs(), rng).rolling(win).mean()


def f_vol_z_20(c, v, o, h, l, m, win=20):
    vv = np.log(v.replace(0, np.nan))
    return (vv - vv.rolling(win).mean()) / vv.rolling(win).std()


def f_mkt_corr_20(c, v, o, h, l, m, win=20):
    spx = m["SPX"].reindex(c.index).ffill().pct_change()
    return c.pct_change().rolling(win).corr(spx)


def f_downside_ratio_20(c, v, o, h, l, m, win=20):
    r = c.pct_change()
    tot = r.rolling(win).std()
    d = r.clip(upper=0).rolling(win).std()
    return _safe_div(d, tot)


def f_updown_freq_20(c, v, o, h, l, m, win=20):
    r = c.pct_change()
    return r.rolling(win).apply(lambda x: (x > 0).sum() / max((x != 0).sum(), 1), raw=True)


fns = {
    "body_ratio_10": f_body_ratio_10,
    "vol_z_20": f_vol_z_20,
    "mkt_corr_20": f_mkt_corr_20,
    "downside_ratio_20": f_downside_ratio_20,
    "updown_freq_20": f_updown_freq_20,
}
panels = {k: factor_panel(fn, close, vol, open_, high, low, macro) for k, fn in fns.items()}

names = list(panels.keys()) + list(lib.keys())
mat = pd.DataFrame(np.nan, index=names, columns=names)
for a in names:
    pa = panels.get(a, lib.get(a))
    for b in names:
        pb = panels.get(b, lib.get(b))
        m = pa.notna() & pb.notna()
        if m.sum().sum() >= 100:
            mat.loc[a, b] = pa[m].stack().rank().corr(pb[m].stack().rank())

print("Pairwise spearman rho (new candidates + library):")
print(mat.round(3).to_string())
print()
for a in names:
    pa = panels.get(a, lib.get(a))
    maxlib = max(abs(pa[mask].stack().rank().corr(lib[k][mask].stack().rank()))
                 for k, libp in lib.items()
                 for mask in [pa.notna() & libp.notna()] if mask.sum().sum() >= 100)
    print(f"{a:20s} max_abs_library_corr = {maxlib:.4f}")
