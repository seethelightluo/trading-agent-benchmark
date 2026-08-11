"""miner2 2027-05-28: diagnostic to reconcile old persisted metrics (ic1=0.0649 n=1171 for nclv_1d)
with fresh re-validation on the same window 2021-01-01..2026-07-15.
Checks: union index vs common-trading-day index, MIN_VALID variants, and by-year IC.
"""
import pandas as pd
import numpy as np
import pickle
from scipy.stats import spearmanr

panel = pickle.load(open("scripts/miner2_panel.pkl", "rb"))
C, O, H, L, V, R, M = panel["close"], panel["open"], panel["high"], panel["low"], panel["vol"], panel["ret"], panel["macro"]
ASSETS = list(C.columns)
lnC = np.log(C)

fdf = pd.DataFrame(index=C.index, columns=ASSETS, dtype=float)
for a in ASSETS:
    fdf[a] = -(C[a] - L[a]) / (H[a] - L[a])

print("panel index n:", len(C.index), "| first:", C.index.min().date(), "| last:", C.index.max().date())
print("weekday-only rows:", C.index.dayofweek.le(4).sum())

def ic_series(fdf, h=1, min_valid=8):
    fr = R.shift(-h)
    out = []
    for i in range(len(fdf)):
        fv, rv = fdf.iloc[i].values, fr.iloc[i].values
        m = np.isfinite(fv) & np.isfinite(rv)
        if m.sum() < min_valid:
            continue
        rho = spearmanr(fv[m], rv[m]).correlation
        out.append(rho if np.isfinite(rho) else np.nan)
    return np.array(out)

for win_name, win in [("2021..2026-07-15", slice("2021-01-01", "2026-07-15")),
                      ("2021..2027-05-27", slice("2021-01-01", None))]:
    sub = fdf.loc[win]
    for mv in (8, 10, 15):
        ic = ic_series(sub, 1, mv)
        ok = np.isfinite(ic)
        print(f"win={win_name} min_valid={mv}: n={int(ok.sum())} ic={np.nanmean(ic[ok]):+.4f} icir={np.nanmean(ic[ok])/np.nanstd(ic[ok]):+.4f}")

# common trading days: dates where all 15 assets have close
common = C.notna().all(axis=1)
print("\ncommon-trading-day rows:", int(common.sum()))
sub_c = fdf.loc["2021-01-01":"2026-07-15"][common.loc["2021-01-01":"2026-07-15"]]
ic_c = ic_series(sub_c, 1, 15)
ok = np.isfinite(ic_c)
print(f"common-days 2021..2026-07-15: n={int(ok.sum())} ic={np.nanmean(ic_c[ok]):+.4f} icir={np.nanmean(ic_c[ok])/np.nanstd(ic_c[ok]):+.4f}")

# weekday-only union index
wd = C.index.dayofweek.le(4)
sub_w = fdf.loc["2021-01-01":"2026-07-15"][wd.loc["2021-01-01":"2026-07-15"]]
ic_w = ic_series(sub_w, 1, 8)
ok = np.isfinite(ic_w)
print(f"weekday-only 2021..2026-07-15: n={int(ok.sum())} ic={np.nanmean(ic_w[ok]):+.4f} icir={np.nanmean(ic_w[ok])/np.nanstd(ic_w[ok]):+.4f}")

# by year on union index
for yr in range(2021, 2028):
    sub = fdf.loc[f"{yr}-01-01":f"{yr}-12-31"]
    ic = ic_series(sub, 1, 8)
    ok = np.isfinite(ic)
    if ok.sum() > 20:
        print(f"{yr}: n={int(ok.sum())} ic={np.nanmean(ic[ok]):+.4f} icir={np.nanmean(ic[ok])/np.nanstd(ic[ok]):+.4f}")
