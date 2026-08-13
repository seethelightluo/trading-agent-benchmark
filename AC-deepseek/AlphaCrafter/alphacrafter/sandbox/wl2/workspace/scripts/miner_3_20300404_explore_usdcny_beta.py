"""miner_3 2030-04-04 exploration: USDCNY conditional beta factor (usdcny_beta_cond_60x20).
Idea: assets whose returns load on USDCNY (CNY depreciation = risk-off channel) * USDCNY 20d move.
USDCNY is an observation-only macro signal NOT yet used in the factor library (DXY/USDJPY/VIX are used).
Motivation: in the current broad risk-off tape (WTI/ETH/XAU down, China small-caps up), CNY sentiment
may discriminate cross-sectionally which assets are hit by China/EM stress.
Gates: |IC|>=0.0070, |ICIR|>=0.0840 (daily paper Spearman vs fwd10).
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_3_20260813_lib import (
    ASSETS, load_asset, to_grid, load_macro, cross_sectional_rank,
    spearman_ic_matrix, summarize, decay_curve, fwd_by_horizon_dict,
    turnover_10d_rank, library_pairwise_corr, coverage_stats,
    HORIZON, MIN_ASSETS,
)

DAYS = 3400
FID = "usdcny_beta_cond_60x20"


def roll_beta_cond(asset_ret, ref_ret, w, minp):
    out = pd.Series(np.nan, index=asset_ret.index)
    a = asset_ret.values.astype(float)
    b = ref_ret.reindex(asset_ret.index).values.astype(float)
    for i in range(w - 1, len(a)):
        seg = slice(i - w + 1, i + 1)
        x = b[seg]; y = a[seg]
        m = ~(np.isnan(x) | np.isnan(y))
        x = x[m]; y = y[m]
        if len(x) < minp or np.std(x) < 1e-12:
            continue
        beta = np.cov(x, y)[0, 1] / np.var(x)
        if np.isfinite(beta):
            out.iloc[i] = beta
    return out


series = {}
for s in ASSETS:
    df = load_asset(s, days=DAYS)
    if df is None or len(df) < 100:
        print("skip", s)
        continue
    close = df["close"].astype(float)
    df = pd.DataFrame({"close": close, "ret": close.pct_change(),
                       "open": df["open"].astype(float), "high": df["high"].astype(float),
                       "low": df["low"].astype(float)})
    series[s] = df
print("assets with data:", sorted(series.keys()))

fwd10 = to_grid({s: df["close"].shift(-HORIZON) / df["close"] - 1.0 for s, df in series.items()})
fwd_by_h = fwd_by_horizon_dict(series, horizons=(1, 2, 3, 5, 10, 20))
dates = np.array(__import__("miner_3_20260813_lib", fromlist=["GRID"]).GRID)

usdcny = load_macro("USDCNY")
print("USDCNY loaded:", usdcny is not None, "rows:", len(usdcny) if usdcny is not None else 0)

panel = {}
for s, df in series.items():
    beta = roll_beta_cond(df["ret"], usdcny.reindex(df.index).pct_change(), 60, 30)
    m = usdcny.reindex(df.index) / usdcny.reindex(df.index).shift(20) - 1.0
    panel[s] = beta * m

mat = to_grid(panel)
rank_mat = cross_sectional_rank(mat)
ics = spearman_ic_matrix(rank_mat, fwd10)
print("n_ic_dates:", len(ics))
s = summarize(ics, dates, FID, HORIZON)
rho_dict, rho_name, max_rho = library_pairwise_corr(mat)
s["max_abs_library_correlation"] = round(max_rho, 4)
s["max_corr_with"] = rho_name
s["turnover_10d_rank"] = round(turnover_10d_rank(rank_mat), 4)
cov, dates_ge8 = coverage_stats(mat)
s["coverage"] = round(cov, 4)
s["dates_ge8_frac"] = round(dates_ge8, 4)
s["decay"] = decay_curve(rank_mat, fwd_by_h)
s["ok"] = bool((abs(s["ic"]) >= 0.0070) and (abs(s["icir"]) >= 0.0840))
s.pop("idx", None); s.pop("icv", None)
print(json.dumps(s, indent=1, default=str))
print("GATE PASS:", s["ok"], "| ic=%.4f icir=%.4f" % (s["ic"], s["icir"]))
