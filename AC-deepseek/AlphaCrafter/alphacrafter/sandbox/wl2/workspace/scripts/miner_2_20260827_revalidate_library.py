"""miner_2 re-validation of library factors (2026-08-27 cycle).
Recomputes each effective factor definition on data visible through 2026-08-26
(per-asset own calendar), IC = daily cross-sectional Spearman vs fwd 10d.
Gates: |IC|>=0.0070, |ICIR|>=0.0840. Reports drift vs previous validation.
"""
import json, sys, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_2_20260827_lib import (ASSETS, GRID, N_GRID, HORIZON, MIN_ASSETS,
                                  load_asset, asset_series, to_grid, load_macro,
                                  safe_div, roll_mean, roll_std, rolling_corr,
                                  rolling_beta, cross_sectional_rank,
                                  spearman_ic_matrix, summarize, decay_curve,
                                  turnover_10d_rank, library_pairwise_corr,
                                  coverage_stats)

print(f"grid: {N_GRID} rows {GRID[0]}..{GRID[-1]}; assets: {ASSETS}")

SERIES = asset_series()
print("assets loaded:", sorted(SERIES.keys()))
FWD10 = to_grid({s: d["fwd10"] for s, d in SERIES.items()})

spx = load_macro("SPX") if os.path.exists("../persistent/index_data/SPX.csv") else None
if spx is None:
    spx = SERIES["SPX"]["close"] if "SPX" in SERIES else None
dxy = load_macro("DXY")
usdjpy = load_macro("USDJPY")
vix = load_macro("VIX")

# macro returns on macro own calendar
def mret(ser):
    if ser is None:
        return None
    return ser.pct_change()

SPX_R = mret(spx)
DXY_R = mret(dxy)
JPY_R = mret(usdjpy)
VIX_R = mret(vix)

# ---------- compute all library factors per asset (own calendar) ----------
F = {s: {} for s in SERIES}

for s, d in SERIES.items():
    close = d["close"].astype(float)
    ret = d["ret"]
    o = None; h = None; l = None; vol = None
    raw = load_asset(s)
    if raw is not None:
        o = raw["open"].astype(float)
        h = raw["high"].astype(float)
        l = raw["low"].astype(float)
        vol = pd.to_numeric(raw["volume"], errors="coerce")
    n = len(close)
    idx = d.index
    # reference returns aligned to asset own dates
    spxr = SPX_R.reindex(idx).values if SPX_R is not None else np.full(n, np.nan)
    dxyv = DXY_R.reindex(idx).values if DXY_R is not None else np.full(n, np.nan)
    jpyv = JPY_R.reindex(idx).values if JPY_R is not None else np.full(n, np.nan)
    vixv = VIX_R.reindex(idx).values if VIX_R is not None else np.full(n, np.nan)
    rv = ret.values.astype(float)

    # 1 calmness_20
    sd20 = roll_std(rv, 20)
    calm = roll_mean((np.abs(rv) < 0.5 * sd20).astype(float), 20)
    F[s]["calmness_20"] = calm

    # 2 close_pos_20
    if o is not None:
        rng = (h.values - l.values)
        cpos = (close.values - l.values) / np.where(np.abs(rng) < 1e-12, np.nan, rng)
        F[s]["close_pos_20"] = roll_mean(cpos, 20)
    else:
        F[s]["close_pos_20"] = np.full(n, np.nan)

    # 3 days_since_high_60
    dsh = np.full(n, np.nan)
    cv = close.values
    for i in range(59, n):
        seg = cv[i - 59:i + 1]
        mx = np.nanmax(seg)
        if not np.isfinite(mx):
            continue
        hits = np.where(seg == mx)[0]
        if len(hits):
            dsh[i] = 59 - hits[-1]
    F[s]["days_since_high_60"] = dsh

    # 4 downbeta_spx_60
    db = np.full(n, np.nan)
    for i in range(59, n):
        a = rv[i - 59:i + 1]; b = spxr[i - 59:i + 1]
        ok = ~(np.isnan(a) | np.isnan(b)) & (b < 0)
        if ok.sum() < 15:
            continue
        aa, bb = a[ok], b[ok]
        if bb.std() < 1e-12:
            continue
        db[i] = np.cov(aa, bb)[0, 1] / np.var(bb)
    F[s]["downbeta_spx_60"] = db

    # 5 dxy_beta_cond_60x20
    if dxyv is not None:
        b60 = rolling_beta(rv, dxyv, 60, 30)
        dxy20 = dxy.reindex(idx).values if dxy is not None else np.full(n, np.nan)
        d20 = np.full(n, np.nan)
        d20[20:] = dxy20[20:] / dxy20[:-20] - 1.0
        F[s]["dxy_beta_cond_60x20"] = b60 * d20
    else:
        F[s]["dxy_beta_cond_60x20"] = np.full(n, np.nan)

    # 6 gain_loss_20
    g = np.where(rv > 0, rv, 0.0); ls = np.where(rv < 0, rv, 0.0)
    F[s]["gain_loss_20"] = safe_div(roll_mean(g, 20), np.abs(roll_mean(ls, 20)) + 1e-9)

    # 7 intraday_drift_20
    if o is not None:
        F[s]["intraday_drift_20"] = roll_mean(close.values / o.values - 1.0, 20)
    else:
        F[s]["intraday_drift_20"] = np.full(n, np.nan)

    # 8 lagbeta_spx_60
    lb = np.full(n, np.nan)
    for i in range(59, n):
        a = rv[i - 59:i + 1]; b = np.concatenate([[np.nan], spxr[i - 60:i]])
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < 30:
            continue
        aa, bb = a[ok], b[ok]
        if bb.std() < 1e-12:
            continue
        lb[i] = np.cov(aa, bb)[0, 1] / np.var(bb)
    F[s]["lagbeta_spx_60"] = lb

    # 9/10 max_consec_gain/loss_20
    mg = np.full(n, np.nan); ml = np.full(n, np.nan)
    ups = (rv > 0).astype(int); dns = (rv < 0).astype(int)
    for i in range(19, n):
        su = ups[i - 19:i + 1]; sd = dns[i - 19:i + 1]
        if su.sum() >= 1:
            # longest consecutive run of 1s
            best = 0; cur = 0
            for v in su:
                cur = cur + 1 if v else 0
                best = max(best, cur)
            mg[i] = best
        if sd.sum() >= 1:
            best = 0; cur = 0
            for v in sd:
                cur = cur + 1 if v else 0
                best = max(best, cur)
            ml[i] = best
    F[s]["max_consec_gain_20"] = mg
    F[s]["max_consec_loss_20"] = ml

    # 11 mom20_volproxy60 (expr: close.shift(5)/close.shift(25)-1)
    F[s]["mom20_volproxy60"] = close.shift(5).values / close.shift(25).values - 1.0

    # 12 mom30_vol60
    m30 = close.shift(5).values / close.shift(35).values - 1.0
    sd60 = roll_std(rv, 60)
    F[s]["mom30_vol60"] = safe_div(m30, sd60)

    # 13-16 momentum skips
    F[s]["mom_10d_skip5"] = close.shift(5).values / close.shift(15).values - 1.0
    F[s]["mom_20d_skip5"] = close.shift(5).values / close.shift(25).values - 1.0
    F[s]["mom_120d_skip5"] = close.shift(5).values / close.shift(125).values - 1.0
    F[s]["mom_180d_skip5"] = close.shift(5).values / close.shift(185).values - 1.0

    # 17 range_pos_252
    rmin = close.rolling(252, min_periods=30).min().values
    rmax = close.rolling(252, min_periods=30).max().values
    F[s]["range_pos_252"] = safe_div(cv - rmin, rmax - rmin)

    # 18 spx_corr60
    F[s]["spx_corr60"] = rolling_corr(rv, spxr, 60, 15)

    # 19 usdjpy_beta_cond_120x60
    if jpyv is not None:
        b120 = rolling_beta(rv, jpyv, 120, 60)
        j60 = np.full(n, np.nan)
        jv = usdjpy.reindex(idx).values if usdjpy is not None else np.full(n, np.nan)
        j60[60:] = jv[60:] / jv[:-60] - 1.0
        F[s]["usdjpy_beta_cond_120x60"] = b120 * j60
    else:
        F[s]["usdjpy_beta_cond_120x60"] = np.full(n, np.nan)

    # 20 vix_beta_cond_60x20
    if vixv is not None:
        bv = rolling_beta(rv, vixv, 60, 30)
        v20 = np.full(n, np.nan)
        vv = vix.reindex(idx).values if vix is not None else np.full(n, np.nan)
        v20[20:] = vv[20:] / vv[:-20] - 1.0
        F[s]["vix_beta_cond_60x20"] = -bv * v20
    else:
        F[s]["vix_beta_cond_60x20"] = np.full(n, np.nan)

    # 21 vol_of_vol20x60
    sd20s = roll_std(rv, 20)
    F[s]["vol_of_vol20x60"] = roll_std(sd20s, 60)

    # 22 volcluster_60
    ar = np.abs(rv)
    F[s]["volcluster_60"] = rolling_corr(ar, np.concatenate([[np.nan], ar[:-1]]), 60, 40)

# ---------- build matrices & evaluate ----------
DATES = np.array(GRID)
results = {}
print(f"\n=== RE-VALIDATION (data through {GRID[-1]}, fwd {HORIZON}d) ===")
for name in sorted(F[SERIES.keys().__iter__().__next__()].keys()):
    mat = to_grid({s: pd.Series(F[s][name], index=SERIES[s].index) for s in SERIES})
    ics = spearman_ic_matrix(mat, FWD10)
    if len(ics) < 200:
        print(f"{name:28s} insufficient IC dates ({len(ics)})")
        continue
    s = summarize(ics, DATES, name, HORIZON)
    cov_ad, cov_d8 = coverage_stats(mat)
    rank_mat = cross_sectional_rank(mat)
    to10 = turnover_10d_rank(rank_mat)
    dec = decay_curve(mat, SERIES)
    corrs, mxname, mxabs = library_pairwise_corr(mat)
    prev_ic = None
    results[name] = {"ic": round(s["ic"], 4), "icir": round(s["icir"], 3),
                     "hit": round(s["hit"], 3), "n": s["n_ic_dates"],
                     "coverage_ad": round(cov_ad, 3), "coverage_d8": round(cov_d8, 3),
                     "turnover_10d": round(to10, 3), "decay": dec,
                     "max_lib_corr": round(mxabs, 3), "max_lib_corr_name": mxname,
                     "regime": s["regime"],
                     "pass_gate": (abs(s["ic"]) >= 0.0070 and abs(s["icir"]) >= 0.0840),
                     "period": f"{GRID[0]}..{GRID[-1]}"}
    print(f"{name:28s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['hit']:.3f} n={s['n_ic_dates']} "
          f"cov={cov_ad:.3f}/{cov_d8:.3f} to10={to10:.3f} maxLibCorr={mxabs:.3f}({mxname}) "
          f"PASS={results[name]['pass_gate']}")
    print(f"  dec={dec}  reg={json.dumps(s['regime'])}")

json.dump(results, open("scripts/miner_2_20260827_revalidation_results.json", "w"), indent=1)
print("\nsaved scripts/miner_2_20260827_revalidation_results.json")
