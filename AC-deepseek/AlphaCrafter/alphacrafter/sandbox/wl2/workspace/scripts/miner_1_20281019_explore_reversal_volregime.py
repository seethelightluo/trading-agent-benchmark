"""miner_1 2028-10-19: explore short-term reversal + vol-regime factor family.
Rationale: memory flags reversal tape (momentum adds whipsawed 4+ blocks) and
VIX stress spike (41.9, +120%/20d); short-horizon mean reversion and
vol-regime-shift avoidance are natural candidates for this tape.
Each candidate validated vs fwd10 (own-calendar) cross-sectional rank IC.
Gates: |IC|>=0.0070, |ICIR|>=0.0840.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_3_20260813_lib import (
    asset_series, to_grid, cross_sectional_rank, spearman_ic_matrix,
    summarize, decay_curve, fwd_by_horizon_dict, turnover_10d_rank,
    library_pairwise_corr, coverage_stats, GRID, HORIZON,
)

series = asset_series()
fwd10 = to_grid({s: df["fwd10"] for s, df in series.items()})
fwd_by_h = fwd_by_horizon_dict(series, horizons=(1, 2, 3, 5, 10, 20))
dates = np.array(GRID)

def rstd(x, w, minp):
    return x.rolling(w, min_periods=minp).std()

def eval_factor(fid, panel):
    mat = to_grid(panel)
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(rank_mat, fwd10)
    if len(ics) == 0:
        print(f"{fid:28s} NO IC DATES"); return None
    s = summarize(ics, dates, fid, HORIZON)
    rho_dict, rho_name, max_rho = library_pairwise_corr(mat)
    s["max_abs_library_correlation"] = round(max_rho, 4)
    s["max_corr_with"] = rho_name
    s["turnover_10d_rank"] = round(turnover_10d_rank(rank_mat), 4)
    cov, dates_ge8 = coverage_stats(mat)
    s["coverage"] = round(cov, 4)
    s["decay"] = decay_curve(rank_mat, fwd_by_h)
    s["ok"] = bool((abs(s["ic"]) >= 0.0070) and (abs(s["icir"]) >= 0.0840))
    l250 = s["regime"].get("last250", {})
    print(f"{fid:28s} ic={s['ic']:+.4f} icir={s['icir']:+.4f} hit={s['hit']:.3f} "
          f"turn={s['turnover_10d_rank']:.3f} cov={s['coverage']:.3f} maxrho={max_rho:.3f} "
          f"last250={l250.get('ic','NA')} ok={s['ok']}")
    s.pop("idx", None); s.pop("icv", None)
    return s

factors = {}

# R1: 5d reversal (skip 1) - short-term mean reversion
for s, df in series.items():
    c = df["close"]
    factors.setdefault("rev5_skip1", {})[s] = -(c.shift(1) / c.shift(6) - 1.0)

# R2: 10d reversal (skip 1)
for s, df in series.items():
    c = df["close"]
    factors.setdefault("rev10_skip1", {})[s] = -(c.shift(1) / c.shift(11) - 1.0)

# R3: vol-normalized 5d reversal (skip 1), 20d vol denom
for s, df in series.items():
    c = df["close"]
    mom = c.shift(1) / c.shift(6) - 1.0
    v = df["ret"].rolling(20, min_periods=10).std()
    factors.setdefault("rev5_vol20", {})[s] = -mom / (v + 1e-9)

# R4: 3d reversal normalized by 20d vol
for s, df in series.items():
    c = df["close"]
    mom = c.shift(1) / c.shift(4) - 1.0
    v = df["ret"].rolling(20, min_periods=10).std()
    factors.setdefault("rev3_vol20", {})[s] = -mom / (v + 1e-9)

# R5: vol-regime shift - 5d realized vol / 60d realized vol (negative => avoid vol surge)
for s, df in series.items():
    rv5 = df["ret"].rolling(5, min_periods=3).std()
    rv60 = df["ret"].rolling(60, min_periods=15).std()
    factors.setdefault("rv_ratio_5x60", {})[s] = -rv5 / (rv60 + 1e-9)

# R6: vol-regime shift 20x60
for s, df in series.items():
    rv20 = df["ret"].rolling(20, min_periods=10).std()
    rv60 = df["ret"].rolling(60, min_periods=15).std()
    factors.setdefault("rv_ratio_20x60", {})[s] = -rv20 / (rv60 + 1e-9)

# R7: max single-day loss in last 10d (penalize recent crash assets) -> negative
for s, df in series.items():
    r = df["ret"]
    factors.setdefault("min_ret_10", {})[s] = -r.rolling(10, min_periods=5).min()

results = {}
for fid, panel in factors.items():
    s = eval_factor(fid, panel)
    if s is not None:
        results[fid] = s

out = "scripts/miner_1_20281019_reversal_volregime_results.json"
json.dump(results, open(out, "w"), indent=1, default=str)
print("DONE", out)
