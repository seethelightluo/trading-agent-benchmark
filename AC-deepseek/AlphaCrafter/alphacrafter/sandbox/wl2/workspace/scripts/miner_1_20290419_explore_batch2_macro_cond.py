"""miner_1 2029-04-19: exploration batch 2 - macro-momentum-conditional cross-asset factors
(corr(asset, ref) x ref momentum) + coiling/gap variants. Motivation: trader flags repeated
whipsaw; cross-asset spillover conditional on macro momentum may add orthogonal timing.
Full window 2020-01-01..2029-04-18. Gates: |IC|>=0.0070, |ICIR|>=0.0840.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_3_20260813_lib import (
    GRID, ASSETS, HORIZON, MIN_ASSETS,
    load_asset, to_grid, load_macro,
    safe_div, cross_sectional_rank, spearman_ic_matrix,
    summarize, decay_curve, fwd_by_horizon_dict, turnover_10d_rank,
    library_pairwise_corr, coverage_stats,
)


def asset_series_full():
    out = {}
    for s in ASSETS:
        df = load_asset(s, days=2600)
        if df is None or len(df) < 100:
            continue
        close = df["close"].astype(float)
        ret = close.pct_change()
        d = pd.DataFrame({
            "close": close, "ret": ret, "fwd10": close.shift(-HORIZON) / close - 1.0,
            "open": df["open"].astype(float), "high": df["high"].astype(float),
            "low": df["low"].astype(float), "volume": df["volume"].astype(float),
        })
        out[s] = d
    return out


def roll_corr(a, b, w, minp=15):
    return a.rolling(w, min_periods=minp).corr(b)


def mom(x, w):
    return x / x.shift(w) - 1.0


series = asset_series_full()
print("assets:", sorted(series.keys()))
fwd10 = to_grid({s: df["fwd10"] for s, df in series.items()})
fwd_by_h = fwd_by_horizon_dict(series, horizons=(1, 2, 3, 5, 10, 20))
dates = np.array(GRID)

spx = series["SPX"]["close"]
dxy = load_macro("DXY")
vix = load_macro("VIX")
usdjpy = load_macro("USDJPY")

refs = {
    "spx": spx,
    "btc": series["BTC"]["close"],
    "us10y": series["US10Y"]["close"],
    "wti": series["WTI"]["close"],
    "copper": series["COPPER"]["close"],
    "eth": series["ETH"]["close"],
}
for name, ser in [("dxy", dxy), ("vix", vix), ("usdjpy", usdjpy)]:
    if ser is not None:
        refs[name] = ser

mom_windows = {"20": 20, "60": 60}
factors = {}

for ref_name, ref_close in refs.items():
    if ref_close is None:
        continue
    ref_ret = ref_close.pct_change()
    for mw, w in mom_windows.items():
        ref_m = mom(ref_close, w)
        fid = f"corr_{ref_name}_mom{mw}"
        for s, df in series.items():
            c = roll_corr(df["ret"], ref_ret.reindex(df.index), 60, min_periods=20)
            factors.setdefault(fid, {})[s] = c * ref_m.reindex(df.index)

# coiling: 20d mean of (high-low)/close, negated (squeeze -> expansion)
for s, df in series.items():
    rng = (df["high"] - df["low"]) / df["close"]
    factors.setdefault("coiling_20", {})[s] = -rng.rolling(20, min_periods=10).mean()

# gap_drift_20: mean overnight gap (open/prev_close - 1) over 20d
for s, df in series.items():
    gap = df["open"] / df["close"].shift(1) - 1.0
    factors.setdefault("gap_drift_20", {})[s] = gap.rolling(20, min_periods=10).mean()

# downside range share: mean((close-low)/(high-low)) vs close_pos - use (low to close) drift variant
for s, df in series.items():
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    upper_shadow = (df["high"] - df["close"]) / rng
    factors.setdefault("upper_shadow_20", {})[s] = -upper_shadow.rolling(20, min_periods=10).mean()

results = {}
for fid, panel in factors.items():
    mat = to_grid(panel)
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(rank_mat, fwd10)
    if len(ics) == 0:
        print(fid, "NO IC"); continue
    s = summarize(ics, dates, fid, HORIZON)
    if s is None:
        continue
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
    results[fid] = s
    reg250 = s["regime"].get("last250", {})
    print(f"{fid:24s} ic={s['ic']:+.4f} icir={s['icir']:+.4f} hit={s['hit']:.3f} "
          f"turn={s['turnover_10d_rank']:.3f} cov={s['coverage']:.3f} maxrho={max_rho:.3f} "
          f"l250_ic={reg250.get('ic','NA')} ok={s['ok']}", flush=True)

json.dump(results, open("scripts/miner_1_20290419_batch2_results.json", "w"), indent=1, default=str)
print("DONE batch2")
