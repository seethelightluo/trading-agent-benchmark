"""miner_1 2032-09-30 diagnostic for gold_vs_rate_60 (missing from 09-16 screen output).

Goal: recompute the candidate exactly as in miner_1_20320916_screen_beta_path.py and
understand why it produced no result entry (exception vs no valid IC dates).
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_3_20260813_lib import (
    ASSETS, load_asset, load_macro, to_grid, cross_sectional_rank,
    spearman_ic_matrix, summarize, decay_curve, fwd_by_horizon_dict,
    turnover_10d_rank, library_pairwise_corr, coverage_stats,
    HORIZON, MIN_ASSETS, GRID, N_GRID,
)

DAYS = 4200
dates = np.array(GRID)

series = {}
for s in ASSETS:
    df = load_asset(s, days=DAYS)
    if df is None or len(df) < 120:
        print("skip", s, flush=True)
        continue
    close = df["close"].astype(float)
    d = pd.DataFrame({
        "close": close, "ret": close.pct_change(),
        "high": df["high"].astype(float), "low": df["low"].astype(float),
        "volume": df["volume"].astype(float),
    })
    series[s] = d
print("assets with data:", len(series), sorted(series.keys()), flush=True)

# gold_vs_rate_60 exactly as in original screen
try:
    panel = {}
    xau = series["XAU"]["close"]
    us10y = series["US10Y"]["close"]
    print("XAU close head:", xau.head(3).values, "tail:", xau.tail(3).values, flush=True)
    print("US10Y close head:", us10y.head(3).values, "tail:", us10y.tail(3).values, flush=True)
    xau_ret60 = xau / xau.shift(60) - 1.0
    us10y_chg60 = us10y - us10y.shift(60)
    spread_asset = xau_ret60 - us10y_chg60
    print("spread_asset describe:", spread_asset.describe().to_dict(), flush=True)
    print("spread_asset NaN frac: %.3f" % spread_asset.isna().mean(), flush=True)
    for s, df in series.items():
        if s == "XAU":
            panel[s] = pd.Series(np.nan, index=df.index)
        else:
            panel[s] = spread_asset.reindex(df.index)
    mat = to_grid(panel)
    cov, dates_ge8 = coverage_stats(mat)
    print("coverage=%.4f dates_ge8=%.4f" % (cov, dates_ge8), flush=True)
    ics = spearman_ic_matrix(mat, to_grid({s: df["close"].shift(-HORIZON) / df["close"] - 1.0 for s, df in series.items()}))
    print("n raw IC obs:", len(ics), flush=True)
    summ = summarize(ics, dates, "gold_vs_rate_60", HORIZON)
    if summ is None:
        print("RESULT: NO VALID IC DATES", flush=True)
    else:
        print("ic=%.4f icir=%.4f hit=%.3f n=%d" % (summ["ic"], summ["icir"], summ["hit"], summ["n_ic_dates"]), flush=True)
        print("regime:", json.dumps(summ["regime"]), flush=True)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("RESULT: EXCEPTION RAISED -> candidate never validated", flush=True)
