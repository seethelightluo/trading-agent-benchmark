"""miner_3 2026-09-10 exploration batch A: OHLC intraday-geometry + volatility-structure factors.

Motivation: existing library covers momentum (mom_*), trend position (range_pos_252,
days_since_high_60, close_pos_20), vol level/cluster (volcluster_60, vol_of_vol20x60),
and macro beta (downbeta_spx_60, dxy/vix/usdjpy_beta). Little uses candle geometry
(body/shadow structure), gap behaviour, or vol term-structure ratios. These use
open/high/low/close (available for all 15 assets) -> naturally low library correlation.

Candidates (per-asset own calendar):
 1. body_ratio_20        : mean(|close-open| / (high-low), 20)  - candle body dominance
 2. shadow_asym_20       : mean((upper-lower) / (high-low), 20) - wick asymmetry (signed)
 3. lower_shadow_20      : mean((min(o,c)-low)/(high-low), 20)  - demand/support wick size
 4. gap_std_20           : std(open/prev_close - 1, 20)          - gap volatility
 5. gap_abs_mean_20      : mean(|open/prev_close - 1|, 20)       - gap intensity
 6. vol_ratio_20x60      : vol20 / vol60                          - vol regime expansion
 7. range_ratio_20x60    : mean daily range 20 / mean daily range 60
 8. updown_vol_ratio_20  : std(pos rets)/std(neg rets) over 20    - vol asymmetry
 9. downside_freq_20     : fraction of down days over 20         - win-rate mirror
10. park_ratio_20        : parkinson_vol20 / cc_vol20            - intraday vs close-to-close
11. kurt_60              : excess kurtosis of 60d returns         - tail heaviness
12. max_down_day_20      : most negative daily return over 20d   - downside tail

Gates: |IC|>=0.007, |ICIR|>=0.084 on daily cross-sectional Spearman vs fwd 10d.
Library correlation gate: target max |rho| < 0.5 (audit contract threshold).
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
from miner_3_20260813_lib import (ASSETS, GRID, HORIZON, to_grid,
                                  cross_sectional_rank, spearman_ic_matrix,
                                  summarize, decay_curve, fwd_by_horizon_dict,
                                  turnover_10d_rank, library_pairwise_corr,
                                  coverage_stats, safe_div)


def load_asset(sym, days=2300):
    df = get_stock_daily_data(sym, days=days)
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ret"] = df["close"].pct_change()
    df["gap"] = df["open"] / df["close"].shift(1) - 1.0
    df["rng"] = df["high"] - df["low"]
    df["rng_pct"] = df["rng"] / df["close"]
    df["body"] = (df["close"] - df["open"]).abs()
    df["upper"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower"] = df[["open", "close"]].min(axis=1) - df["low"]
    return df


series = {s: load_asset(s) for s in ASSETS}
series = {s: df for s, df in series.items() if df is not None and len(df) > 100}
print(f"assets loaded: {len(series)}/15 -> {sorted(series.keys())}")
fwd = fwd_by_horizon_dict(series)
dates = np.array(GRID)
results = {}


def report(name, mat):
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(mat, fwd[10])
    summ = summarize(ics, dates, name, HORIZON)
    if summ is None:
        print(name, "NO VALID IC DATES")
        return None
    cov_ad, cov_d8 = coverage_stats(mat)
    to = turnover_10d_rank(rank_mat)
    dec = decay_curve(mat, fwd)
    corrs, mx_name, mx_abs = library_pairwise_corr(mat)
    ic, icir = summ["ic"], summ["icir"]
    ok = (abs(ic) >= 0.007) and (abs(icir) >= 0.084)
    corr_ok = mx_abs < 0.5
    print("=" * 90)
    print(f"{name}: IC={ic:+.4f} ICIR={icir:+.4f} hit={summ['hit']:.3f} n={summ['n_ic_dates']} "
          f"cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} turn={to:.3f} maxlibcorr={mx_abs:.3f} ({mx_name}) "
          f"GATE={ok} CORR_OK={corr_ok}")
    print("   regime:", {k: v for k, v in summ["regime"].items()})
    print("   decay:", dec)
    results[name] = {
        "ic": round(ic, 5), "icir": round(icir, 5), "hit": round(summ["hit"], 4),
        "n_ic_dates": summ["n_ic_dates"], "regime": summ["regime"],
        "coverage_asset_days": round(cov_ad, 4), "coverage_dates_ge8": round(cov_d8, 4),
        "turnover_10d_rank": round(to, 4), "decay": dec,
        "max_abs_library_correlation": round(mx_abs, 4),
        "max_lib_corr_name": mx_name, "pass_gate": bool(ok), "corr_ok": bool(corr_ok),
    }
    return summ


def mean_series(df, expr, w, minp):
    s = pd.Series(np.asarray(expr).ravel(), index=df.index)
    return s.rolling(w, min_periods=minp).mean()


def std_series(df, expr, w, minp):
    s = pd.Series(np.asarray(expr).ravel(), index=df.index)
    return s.rolling(w, min_periods=minp).std()


# 1. body_ratio_20
cand = {}
for s, df in series.items():
    cand[s] = mean_series(df, safe_div(df["body"], df["rng"]), 20, 10)
report("body_ratio_20", to_grid(cand))

# 2. shadow_asym_20 (signed wick asymmetry)
cand = {}
for s, df in series.items():
    cand[s] = mean_series(df, safe_div(df["upper"] - df["lower"], df["rng"]), 20, 10)
report("shadow_asym_20", to_grid(cand))

# 3. lower_shadow_20
cand = {}
for s, df in series.items():
    cand[s] = mean_series(df, safe_div(df["lower"], df["rng"]), 20, 10)
report("lower_shadow_20", to_grid(cand))

# 4. gap_std_20
cand = {}
for s, df in series.items():
    cand[s] = std_series(df, df["gap"], 20, 10)
report("gap_std_20", to_grid(cand))

# 5. gap_abs_mean_20
cand = {}
for s, df in series.items():
    cand[s] = mean_series(df, df["gap"].abs(), 20, 10)
report("gap_abs_mean_20", to_grid(cand))

# 6. vol_ratio_20x60
cand = {}
for s, df in series.items():
    v20 = df["ret"].rolling(20, min_periods=10).std()
    v60 = df["ret"].rolling(60, min_periods=30).std()
    cand[s] = pd.Series(safe_div(v20, v60), index=df.index)
report("vol_ratio_20x60", to_grid(cand))

# 7. range_ratio_20x60
cand = {}
for s, df in series.items():
    r20 = df["rng_pct"].rolling(20, min_periods=10).mean()
    r60 = df["rng_pct"].rolling(60, min_periods=30).mean()
    cand[s] = pd.Series(safe_div(r20, r60), index=df.index)
report("range_ratio_20x60", to_grid(cand))

# 8. updown_vol_ratio_20
cand = {}
for s, df in series.items():
    pos = df["ret"].where(df["ret"] > 0)
    neg = df["ret"].where(df["ret"] < 0)
    sp = pos.rolling(20, min_periods=8).std()
    sn = neg.rolling(20, min_periods=8).std()
    cand[s] = pd.Series(safe_div(sp, sn), index=df.index)
report("updown_vol_ratio_20", to_grid(cand))

# 9. downside_freq_20
cand = {}
for s, df in series.items():
    dn = (df["ret"] < 0).astype(float)
    cand[s] = pd.Series(dn.rolling(20, min_periods=10).mean(), index=df.index)
report("downside_freq_20", to_grid(cand))

# 10. park_ratio_20: parkinson vol vs close-to-close vol
cand = {}
for s, df in series.items():
    hl = np.log(df["high"] / df["low"])
    park = np.sqrt((hl ** 2).rolling(20, min_periods=10).mean() / (4.0 * np.log(2.0)))
    cc = df["ret"].rolling(20, min_periods=10).std()
    cand[s] = pd.Series(safe_div(park, cc), index=df.index)
report("park_ratio_20", to_grid(cand))

# 11. kurt_60 (excess kurtosis of returns)
cand = {}
for s, df in series.items():
    r = df["ret"]
    mu = r.rolling(60, min_periods=40).mean()
    sd = r.rolling(60, min_periods=40).std()
    m4 = (r ** 4).rolling(60, min_periods=40).mean()
    kurt = safe_div(m4, sd ** 4) - 3.0
    cand[s] = pd.Series(kurt, index=df.index)
report("kurt_60", to_grid(cand))

# 12. max_down_day_20 (most negative daily return)
cand = {}
for s, df in series.items():
    cand[s] = pd.Series(df["ret"].rolling(20, min_periods=10).min(), index=df.index)
report("max_down_day_20", to_grid(cand))

json.dump(results, open("scripts/miner_3_20260910_batchA_results.json", "w"), indent=1)
print("SAVED batchA results")
