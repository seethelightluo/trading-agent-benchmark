"""miner_3 rigorous validation 2026-07-30.

Top candidates from screen v2/v3: full ic_analysis (horizon 10 admission),
decay, turnover, coverage, pairwise redundancy among candidates, and
max-abs correlation vs the rebuilt old 4-factor library (provenance audit).

Uses the shared API-based factor_validation_lib so metrics match the gate.
Data visible through 2026-07-29 (previous completed trading day).
"""
import sys
sys.path.insert(0, "scripts")
import json
import numpy as np
import pandas as pd
from factor_validation_lib import (TRADABLE, MIN_INSTR, load_panel, load_macro,
                                   rank_ic_series, ic_analysis, library_corr,
                                   align_fwd_returns)

MAXD = "2026-07-29"
panel = load_panel(max_date=MAXD)
ret = panel.pct_change()
print(f"panel dates: {len(panel)}, instruments: {len(panel.columns)}")


def roll_std(x, w):
    return x.rolling(w, min_periods=max(10, w // 2)).std()


def roll_mean(x, w):
    return x.rolling(w, min_periods=max(10, w // 2)).mean()


def rsum(x, w):
    return x.rolling(w, min_periods=max(10, w // 2)).sum()


# ---------------- candidate signal definitions ----------------
C = {}
# momentum family
C["mom_20d_skip5"] = panel.shift(5) / panel.shift(25) - 1.0
C["risk_adj_mom_20d"] = (panel.shift(5) / panel.shift(25) - 1.0) / roll_std(ret, 20)
C["rel_strength_120d"] = (panel / roll_mean(panel, 120)) - 1.0
# VIX / macro risk family
vix = load_macro("VIX", max_date=MAXD)
vixr = vix.pct_change()
beta_vix = ret.rolling(60, min_periods=30).cov(vixr) / vixr.rolling(60, min_periods=30).var()
C["vix_beta_60d"] = beta_vix
corr_vix = ret.rolling(60, min_periods=30).corr(vixr)
C["corr_vix_60d"] = corr_vix
C["vix_cond_60x20"] = -beta_vix * (vix / vix.shift(20) - 1.0)
dxy = load_macro("DXY", max_date=MAXD)
dxy_r = dxy.pct_change()
beta_dxy = ret.rolling(120, min_periods=60).cov(dxy_r) / dxy_r.rolling(120, min_periods=60).var()
C["dxy_beta_120d"] = beta_dxy
# cross-asset market beta
ew = panel.mean(axis=1)
ewr = ew.pct_change()
C["beta_ew_60d"] = ret.rolling(60, min_periods=30).cov(ewr) / ewr.rolling(60, min_periods=30).var()
# intraday / volume / distribution family
high = None  # need OHLC for some; use panel-only where possible

def _hlc_family():
    out = {}
    # win rate: fraction of up days over 20d
    up = (ret > 0).astype(float)
    out["win_rate_20d"] = rsum(up, 20)
    # up/down asymmetry: mean up-day ret / mean down-day ret magnitude
    upm = ret.where(ret > 0, 0.0)
    dnm = ret.where(ret < 0, 0.0)
    out["updown_asym_20d"] = (rsum(upm, 20) / rsum(up, 20)) / ((-rsum(dnm, 20)) / rsum((ret < 0).astype(float), 20))
    # max daily return over 20d (lottery)
    out["max_ret_20d"] = ret.rolling(20, min_periods=10).max()
    # vol ratio short/long
    out["vol_ratio_5x60"] = ret.rolling(5, min_periods=3).std() / roll_std(ret, 60)
    return out


C.update(_hlc_family())

# price-volume correlation needs volume: load via API-free CSV panel in factor_utils
sys.path.insert(0, "scripts")
from factor_utils import load_panel as csv_panel, CURRENT_DATE
px_csv, vol_csv = csv_panel()
px_csv = px_csv[px_csv.index <= pd.Timestamp(MAXD)]
vol_csv = vol_csv[vol_csv.index <= pd.Timestamp(MAXD)]
ret_csv = px_csv.pct_change()
pc = (ret_csv.rolling(20, min_periods=10).corr(vol_csv.pct_change().rolling(1).mean().replace(0, np.nan)))
# align to API panel index
pc_al = pc.reindex(panel.index)
C["price_vol_corr_20d"] = pc_al

print(f"candidate count: {len(C)}")
names = list(C.keys())
for n in names:
    C[n] = C[n].reindex(panel.index)

# ---------------- admission metrics (horizon 10) ----------------
print("\n=== ADMISSION (horizon=10) ===")
print(f"{'factor':<22}{'ic':>7}{'icir':>8}{'hit':>6}{'n':>6}  gate")
adm = {}
for n in names:
    r = ic_analysis(C[n], panel, horizon=10, label=n)
    adm[n] = r
    ok = abs(r["ic"]) >= 0.007 and abs(r["icir"]) >= 0.084
    print(f"{n:<22}{r['ic']:>7.4f}{r['icir']:>8.4f}{r['ic_hit_ratio']:>6.3f}{r['n_ic_dates']:>6d}  {'PASS' if ok else ''}")

# ---------------- pairwise redundancy among candidates ----------------
def daily_mean_rho(a, b):
    """Mean of daily cross-sectional Spearman rho between two signals."""
    rs = []
    fa = a.rank(axis=1, pct=True)
    fb = b.rank(axis=1, pct=True)
    for d in fa.index.intersection(fb.index):
        x, y = fa.loc[d], fb.loc[d]
        m = x.notna() & y.notna()
        if m.sum() >= MIN_INSTR and x[m].nunique() > 1 and y[m].nunique() > 1:
            rho = x[m].corr(y[m], method="spearman")
            if rho is not None and np.isfinite(rho):
                rs.append(rho)
    return float(np.mean(rs)) if rs else float("nan")


print("\n=== PAIRWISE RHO (candidates) ===")
K = len(names)
rho_mat = pd.DataFrame(np.nan, index=names, columns=names)
for i in range(K):
    for j in range(i + 1, K):
        r = daily_mean_rho(C[names[i]], C[names[j]])
        rho_mat.iloc[i, j] = rho_mat.iloc[j, i] = r
print(rho_mat.round(3).to_string())
print("\nPairs with |rho| >= 0.60:")
for i in range(K):
    for j in range(i + 1, K):
        r = rho_mat.iloc[i, j]
        if abs(r) >= 0.60:
            print(f"  {names[i]} ~ {names[j]}: rho={r:.3f}")

# ---------------- library correlation (provenance audit) ----------------
old_lib = {}
old_lib["mom_10d_skip5"] = panel.shift(5) / panel.shift(15) - 1.0
old_lib["mom_120d_skip5"] = panel.shift(5) / panel.shift(125) - 1.0
old_lib["vol_of_vol20x60"] = ret.rolling(20).std().rolling(60).std()
vixr2 = vix.pct_change()
b60 = ret.rolling(60, min_periods=30).cov(vixr2) / vixr2.rolling(60, min_periods=30).var()
old_lib["vix_beta_cond_60x20"] = -b60 * (vix / vix.shift(20) - 1.0)

print("\n=== MAX ABS LIBRARY CORRELATION ===")
lib_corr = {}
for n in names:
    lc = library_corr(C[n], old_lib)
    lib_corr[n] = lc
    print(f"  {n:<22} max_abs_lib_corr={lc:.3f}")

# ---------------- recent stability (sub-periods) ----------------
print("\n=== SUB-PERIOD STABILITY (horizon=10 IC / ICIR) ===")
for n in names:
    row = [n]
    for lo, hi in [("2020-01-01", "2022-12-31"), ("2023-01-01", MAXD), ("2024-01-01", MAXD)]:
        sub = panel.loc[lo:hi]
        fac = C[n].reindex(sub.index)
        r = ic_analysis(fac, sub, horizon=10, label=n)
        row.append(f"{r['ic']:+.3f}/{r['icir']:+.2f}")
    print("  " + "  ".join(row))

# save results for persistence step
out = {"admission": {n: adm[n] for n in names},
       "lib_corr": lib_corr,
       "pairwise_rho": rho_mat.round(4).to_dict()}
with open("scripts/miner_3_validation_top_results.json", "w") as f:
    json.dump(out, f, indent=1, default=str)
print("\nsaved results -> scripts/miner_3_validation_top_results.json")
