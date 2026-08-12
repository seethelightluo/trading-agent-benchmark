"""miner_1 2029-06-28: explore batch 1 - fresh candidate factors.
Data through visible date (2029-06-27). Full window 2020-01-01 -> visible.
Gates: |IC| >= 0.0070 AND |ICIR| >= 0.0840 (10d horizon, daily cross-sectional Spearman).
Regime context: persistent reversal tape (last-250 IC of momentum factors negative).
Candidates target: (a) vol-conditioned short-term reversal, (b) vol-slope/compression,
(c) downside asymmetry, (d) drawdown depth, (e) SPX correlation change, (f) overnight gap,
(g) return skewness.
"""
import sys, json, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from miner_3_20260813_lib import (
    GRID, ASSETS, load_asset, to_grid, load_macro,
    safe_div, cross_sectional_rank, spearman_ic_matrix,
    summarize, decay_curve, fwd_by_horizon_dict, turnover_10d_rank,
    library_pairwise_corr, coverage_stats, HORIZON, MIN_ASSETS,
)

OUT = "scripts/miner_1_20290628_batch1_results.json"


def asset_series_full():
    out = {}
    for s in ASSETS:
        df = load_asset(s, days=3400)
        if df is None or len(df) < 100:
            continue
        close = df["close"].astype(float)
        ret = close.pct_change()
        fwd = close.shift(-HORIZON) / close - 1.0
        d = pd.DataFrame({
            "close": close, "ret": ret, "fwd10": fwd,
            "open": df["open"].astype(float), "high": df["high"].astype(float),
            "low": df["low"].astype(float), "volume": df["volume"].astype(float),
        })
        out[s] = d
    return out


series = asset_series_full()
print("assets with data:", sorted(series.keys()), "n =", len(series))
fwd10 = to_grid({s: df["fwd10"] for s, df in series.items()})
fwd_by_h = fwd_by_horizon_dict(series, horizons=(1, 2, 3, 5, 10, 20))
dates = np.array(GRID)

spx = series["SPX"]["close"] if "SPX" in series else None

factors = {}

# A: vol-conditioned 10d reversal: -ret10 * (vol20/vol60)  (reversal weighted by vol expansion)
for s, df in series.items():
    r = df["ret"]
    v20 = r.rolling(20, min_periods=10).std()
    v60 = r.rolling(60, min_periods=20).std()
    ret10 = df["close"] / df["close"].shift(10) - 1.0
    vol_ratio = v20 / v60.replace(0, np.nan)
    factors.setdefault("rev10_volcond", {})[s] = -ret10 * vol_ratio

# B: vol slope 20/60 - 1 (vol expansion)
for s, df in series.items():
    r = df["ret"]
    v20 = r.rolling(20, min_periods=10).std()
    v60 = r.rolling(60, min_periods=20).std()
    factors.setdefault("vol_slope_20_60", {})[s] = v20 / v60.replace(0, np.nan) - 1.0

# C: downside ratio 20: downside_dev / (downside_dev + upside_dev)
for s, df in series.items():
    r = df["ret"]
    dret = r[r < 0]
    uret = r[r > 0]
    dd = (dret ** 2).rolling(20, min_periods=10).mean().apply(np.sqrt)
    du = (uret ** 2).rolling(20, min_periods=10).mean().apply(np.sqrt)
    factors.setdefault("downside_ratio_20", {})[s] = dd / (dd + du).replace(0, np.nan)

# D: drawdown depth 60: close/rolling_max(close,60) - 1
for s, df in series.items():
    roll_max = df["close"].rolling(60, min_periods=20).max()
    factors.setdefault("drawdown_60", {})[s] = df["close"] / roll_max - 1.0

# E: SPX correlation delta: corr20(asset,SPX) - corr60(asset,SPX)
if spx is not None:
    spx_ret = spx.pct_change()
    for s, df in series.items():
        r = df["ret"]
        c20 = r.rolling(20, min_periods=10).corr(spx_ret)
        c60 = r.rolling(60, min_periods=20).corr(spx_ret)
        factors.setdefault("corr_delta_spx_20_60", {})[s] = c20 - c60

# F: range compression 10/60: (high-low)/close vol proxy 10d vs 60d
for s, df in series.items():
    rng = (df["high"] - df["low"]) / df["close"]
    c10 = rng.rolling(10, min_periods=5).mean()
    c60 = rng.rolling(60, min_periods=20).mean()
    factors.setdefault("range_compress_10_60", {})[s] = c10 / c60.replace(0, np.nan)

# G: overnight gap mean 20: mean(open/prev_close - 1) over 20d
for s, df in series.items():
    gap = df["open"] / df["close"].shift(1) - 1.0
    factors.setdefault("overnight_gap_20", {})[s] = gap.rolling(20, min_periods=10).mean()

# H: return skewness 20
for s, df in series.items():
    factors.setdefault("ret_skew_20", {})[s] = df["ret"].rolling(20, min_periods=10).skew()

results = {}
for label, fac in factors.items():
    mat = to_grid(fac)
    cov, d8 = coverage_stats(mat)
    ics = spearman_ic_matrix(mat, fwd10)
    summ = summarize(ics, dates, label, HORIZON)
    if summ is None:
        print(label, "NO IC DATA")
        continue
    turn = turnover_10d_rank(cross_sectional_rank(mat))
    dec = decay_curve(mat, fwd_by_h)
    corrs, max_name, max_abs = library_pairwise_corr(mat)
    gate_ic = abs(summ["ic"]) >= 0.0070
    gate_icir = abs(summ["icir"]) >= 0.0840
    ok = gate_ic and gate_icir
    results[label] = {
        "label": label, "horizon": HORIZON, "n_ic_dates": summ["n_ic_dates"],
        "ic": summ["ic"], "icir": summ["icir"], "hit": summ["hit"],
        "regime": summ["regime"], "turnover_10d_rank": turn,
        "coverage": round(cov, 4), "dates_ge8_frac": round(d8, 4),
        "decay": dec, "max_abs_library_correlation": round(max_abs, 4),
        "max_corr_with": max_name, "gate_ic": gate_ic, "gate_icir": gate_icir,
        "ok": ok,
    }
    print("=" * 100)
    print("%-28s ic=%+.4f icir=%+.4f hit=%.3f n=%d turn=%.3f cov=%.3f d8=%.3f maxcorr=%.3f(%s) OK=%s"
          % (label, summ["ic"], summ["icir"], summ["hit"], summ["n_ic_dates"], turn, cov, d8, max_abs, max_name, ok))
    for k, v in summ["regime"].items():
        print("    %-10s ic=%+.4f icir=%+.3f n=%d" % (k, v["ic"], v["icir"], v["n"]))
    print("    decay:", {k: v for k, v in dec.items()})

json.dump(results, open(OUT, "w"), indent=1)
print("\nSaved ->", OUT)
print("PASS:", [k for k, v in results.items() if v["ok"]])
