"""miner_3 cycle-2026-08-13: return-distribution risk family v2 (per-asset own calendar).
skew_60, kurt_60, coskew_spx_60. Crash-risk premium hypothesis.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_3_20260813_lib import (GRID, N_GRID, ASSETS, HORIZON, MIN_ASSETS, asset_series,
                                  to_grid, roll_mean, roll_std, safe_div, cross_sectional_rank,
                                  spearman_ic_matrix, summarize, decay_curve, fwd_by_horizon_dict,
                                  turnover_10d_rank, library_pairwise_corr, coverage_stats)

SER = asset_series()
T = N_GRID

def roll_skew(x, w):
    mu = roll_mean(x, w)
    sd = roll_std(x, w)
    m3 = roll_mean(x ** 3, w)
    c3 = np.full(len(x), np.nan)
    c3[w:] = m3[w:] - 3 * mu[w:] * roll_mean(x * x, w)[w:] + 2 * mu[w:] ** 3
    out = np.full(len(x), np.nan)
    ok = sd > 1e-12
    out[ok] = c3[ok] / (sd[ok] ** 3)
    return out

def roll_kurt(x, w):
    mu = roll_mean(x, w)
    sd = roll_std(x, w)
    c4 = roll_mean(x ** 4, w) - 4 * mu * roll_mean(x ** 3, w) + 6 * mu ** 2 * roll_mean(x * x, w) - 3 * mu ** 4
    out = np.full(len(x), np.nan)
    ok = sd > 1e-12
    out[ok] = c4[ok] / (sd[ok] ** 4)
    return out

def roll_coskew(x, m, w):
    mux = roll_mean(x, w); mum = roll_mean(m, w)
    sd_x = roll_std(x, w); sd_m = roll_std(m, w)
    cross = roll_mean((x - mux) * (m - mum) ** 2, w)
    out = np.full(len(x), np.nan)
    ok = (sd_x > 1e-12) & (sd_m > 1e-12)
    out[ok] = cross[ok] / (sd_x[ok] * sd_m[ok] ** 2)
    return out

W = 60
spx = SER["SPX"]
F = {}
for s, df in SER.items():
    r = df["ret"].values.astype(float)
    m = spx["ret"].reindex(df.index).values.astype(float)
    F.setdefault("skew_60", {})[s] = pd.Series(roll_skew(r, W), index=df.index)
    F.setdefault("kurt_60", {})[s] = pd.Series(roll_kurt(r, W), index=df.index)
    F.setdefault("coskew_spx_60", {})[s] = pd.Series(roll_coskew(r, m, W), index=df.index)

FM = {k: to_grid(v) for k, v in F.items()}
fwd10 = to_grid({s: df["fwd10"] for s, df in SER.items()})
fwd_all = fwd_by_horizon_dict(SER)
dates = np.array(GRID)

print(f"grid rows {T} ({GRID[0]}..{GRID[-1]}), assets {len(SER)}, horizon {HORIZON}")
print(f"{'factor':16s} {'IC':>8s} {'ICIR':>7s} {'hit':>5s} {'n':>6s} {'covA':>6s} {'covD8':>6s} {'turn':>6s} {'maxLibRho':>9s}")
results = {}
for name, mat in FM.items():
    ics = spearman_ic_matrix(mat, fwd10)
    s = summarize(ics, dates, name, HORIZON)
    if s is None:
        print(f"{name:16s} NO IC DATES")
        continue
    cov_a, cov_d8 = coverage_stats(mat)
    turn = turnover_10d_rank(cross_sectional_rank(mat))
    libcorr, mxname, mxabs = library_pairwise_corr(mat)
    s["coverage_asset_days"] = round(cov_a, 4)
    s["coverage_dates_ge8"] = round(cov_d8, 4)
    s["turnover_10d_rank"] = round(turn, 4)
    s["max_abs_library_correlation"] = mxabs
    s["max_lib_corr_name"] = mxname
    s["library_pairwise_corr"] = libcorr
    s["decay"] = decay_curve(mat, fwd_all)
    s["pass"] = abs(s["ic"]) >= 0.0070 and abs(s["icir"]) >= 0.0840
    results[name] = s
    print(f"{name:16s} {s['ic']:8.4f} {s['icir']:7.3f} {s['hit']:5.3f} {s['n_ic_dates']:6d} "
          f"{cov_a:6.3f} {cov_d8:6.3f} {turn:6.3f} {mxabs:9.4f}  pass={s['pass']}  (maxcorr={mxname})")
    print("   regime:", {k: v for k, v in s["regime"].items()})
    print("   decay:", s["decay"])

json.dump(results, open("scripts/miner_3_20260813_skew_results.json", "w"), indent=1, default=str)
print("saved scripts/miner_3_20260813_skew_results.json")
