"""miner_2 exploratory sweep: downside/risk/vol-structure factor family.

Validates candidate factors on the 15-asset tradable universe, 2020-01-01..visible-through
(2026-07-29 as current sim date is 2026-07-30; validation window used for admission is
research-only through 2026-07-15 per contract, warm-up ends 2026-07-16 online).

Horizon used: 10d forward returns (admission horizon). Gates: |IC|>=0.0070, |ICIR|>=0.0840.
Also reports decay, coverage, turnover and correlation vs recurring library factors.
"""
from __future__ import annotations
import sys, json
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import miner2_20260730_factorlib as fl

# admission window: study warm-up only (through 2026-07-15)
END = pd.Timestamp("2026-07-15")
START = pd.Timestamp("2020-01-01")

P = fl.load_panel()
P = P[(P.index >= START) & (P.index <= END)]
print(f"panel: {P.shape[0]} dates x {P.shape[1]} assets, {P.index.min().date()}..{P.index.max().date()}")

def std(s, w):
    return s.rolling(w).std()

def candidate_factors(panel):
    out = {}
    close = panel
    for a in panel.columns:
        s = panel[a].dropna()
        out.setdefault('dside_vol_21', {})[a] = -s.pct_change().clip(upper=0).rolling(21).std()
        out.setdefault('skew_60d', {})[a] = s.pct_change().rolling(60).skew()
        out.setdefault('updown_vol_ratio_20', {})[a] = (
            s.pct_change().clip(lower=0).rolling(20).std() /
            (s.pct_change().clip(upper=0).rolling(20).std() + 1e-12))
        out.setdefault('zscore_60d', {})[a] = (s - s.rolling(60).mean()) / (s.rolling(60).std() + 1e-12)
        out.setdefault('mid_range_pos_20d', {})[a] = (
            (s - (s.rolling(20).max() + s.rolling(20).min()) / 2) /
            ((s.rolling(20).max() - s.rolling(20).min()) + 1e-12))
        out.setdefault('efficiency_ratio_20d', {})[a] = (
            (s - s.shift(20)).abs() /
            (s.pct_change().abs().rolling(20).sum() + 1e-12))
        out.setdefault('mdd_depth_60d', {})[a] = s / s.rolling(60).max() - 1.0
        out.setdefault('intraday_share_20d', {})[a] = (
            (s.rolling(20).max() - s.rolling(20).min()) / (s + 1e-12) /
            (s.pct_change().rolling(20).std() + 1e-12))
        out.setdefault('vol_trend_10x30', {})[a] = (
            s.pct_change().rolling(10).std() / (s.pct_change().rolling(30).std() + 1e-12) - 1.0)
        out.setdefault('vol_skew_join', {})[a] = (  # intuition combo: trend-efficiency * (1 - skew)
            (s - s.shift(20)).abs() / (s.pct_change().abs().rolling(20).sum() + 1e-12) *
            (1.0 - s.pct_change().rolling(60).skew().fillna(0.0)))
        out.setdefault('mean_ret_60d', {})[a] = s.pct_change().rolling(60).mean()
        out.setdefault('mean_ret_10d', {})[a] = s.pct_change().rolling(10).mean()
    dfs = {k: pd.DataFrame(v).sort_index() for k, v in out.items()}
    return dfs

F = candidate_factors(P)
fwd10 = fl.fwd_ret_panel(P, 10)

# light correlation baseline vs library factors (recomputed simply)
def simple_lib():
    lib = {}
    for a in P.columns:
        s = P[a].dropna()
        lib.setdefault('mom_10d_skip5', {})[a] = s / s.shift(15) - 1.0
        lib.setdefault('mom_120d_skip5', {})[a] = s / s.shift(125) - 1.0
        lib.setdefault('rng_pos_20d', {})[a] = (s - s.rolling(20).min()) / ((s.rolling(20).max() - s.rolling(20).min()) + 1e-12)
    return {k: pd.DataFrame(v).sort_index() for k, v in lib.items()}

LIB = simple_lib()

rows = []
for name, fv in F.items():
    res = fl.validate(fv, fwd10, label=name, expected_dir=1)
    # negative-direction candidates flip for reporting
    row = {
        "factor": name,
        "ic": res["ic"], "icir": res["icir"], "hit": res["ic_hit_ratio"],
        "coverage": res["coverage"], "turnover": res["turnover_10d_rank"],
        "n_dates": res["n_ic_dates"],
        "decay_5": res["decay_ic_by_horizon"].get("5"),
        "decay_10": res["decay_ic_by_horizon"].get("10"),
        "decay_20": res["decay_ic_by_horizon"].get("20"),
        "passes": res["passes"],
    }
    # max abs library correlation (light proxy: spearman on flattened aligned panel)
    maxrho = 0.0
    for lid, lv in LIB.items():
        both = fv.join(lv, lsuffix='_f', rsuffix='_l')
        c = both.corr(method='spearman')
        diag = np.diag(c.to_numpy())
        # corr matrix cols alternate f/l per asset -> just use pearson flatten on stacked
        s_f = fv.stack(); s_l = lv.stack()
        j = pd.concat([s_f.rename('f'), s_l.rename('l')], axis=1).dropna()
        rho = j['f'].corr(j['l'], method='spearman')
        if np.isfinite(rho):
            maxrho = max(maxrho, abs(rho))
    row["max_abs_lib_rho_light"] = round(maxrho, 3)
    rows.append(row)

df = pd.DataFrame(rows).sort_values("icir", key=lambda x: x.abs(), ascending=False)
print(df.to_string(index=False))