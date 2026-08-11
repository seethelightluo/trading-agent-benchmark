"""miner_3 2026-09-10 exploration batch B: overnight drift, autocorrelation,
short-horizon vol spike, macro-linkage corr (US10Y/XAU/WTI/CN10Y), volume trend,
illiquidity, upside tail, and short-horizon range position.

Motivation: library covers momentum (mom_*), trend position (range_pos_252,
days_since_high_60, close_pos_20, drawup_20), vol level/cluster (volcluster_60,
vol_of_vol20x60, vol_surge_20), skew (ret_skew_10), efficiency (eff_ratio_20),
carry (carry_12m3m/3m1m), and macro betas to SPX/DXY/USDJPY/VIX/crypto. NOT
covered: signed overnight drift, lag-1 return autocorrelation, short-horizon
vol expansion (5/20), bond/gold/energy/CN10Y return correlation, volume trend,
Amihud illiquidity, and max single-day gain.

Gates: |IC|>=0.007, |ICIR|>=0.084 on daily cross-sectional Spearman vs fwd 10d.
Library correlation gate: target max |rho| < 0.5 vs all factors/*.signal.npy.
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

GATE_IC = 0.0070
GATE_ICIR = 0.0840
CORR_LIMIT = 0.5


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
    df["rng_pct"] = (df["high"] - df["low"]) / df["close"]
    return df


series = {s: load_asset(s) for s in ASSETS}
series = {s: df for s, df in series.items() if df is not None and len(df) > 100}
print(f"assets loaded: {len(series)}/15 -> {sorted(series.keys())}")
fwd = fwd_by_horizon_dict(series)
dates = np.array(GRID)
results = {}


def report(name, cand):
    mat = to_grid(cand)
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
    ok = (abs(ic) >= GATE_IC) and (abs(icir) >= GATE_ICIR)
    corr_ok = mx_abs < CORR_LIMIT
    print("=" * 100)
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


def roll_mean_s(s, w, minp):
    return s.rolling(w, min_periods=minp).mean()

def roll_std_s(s, w, minp):
    return s.rolling(w, min_periods=minp).std()


# 1. overnight_drift_20: mean signed overnight gap (open/prev_close - 1)
cand = {s: roll_mean_s(df["gap"], 20, 10) for s, df in series.items()}
report("overnight_drift_20", cand)

# 2. autocorr_20: lag-1 autocorrelation of daily returns over 20d
cand = {}
for s, df in series.items():
    r = df["ret"]
    num = (r * r.shift(1)).rolling(20, min_periods=12).mean() - roll_mean_s(r, 20, 12) * roll_mean_s(r.shift(1), 20, 12)
    den = roll_std_s(r, 20, 12) * roll_std_s(r.shift(1), 20, 12)
    cand[s] = pd.Series(safe_div(num, den), index=df.index)
report("autocorr_20", cand)

# 3. vol_ratio_5x20: short-horizon vol expansion
cand = {}
for s, df in series.items():
    v5 = df["ret"].rolling(5, min_periods=4).std()
    v20 = df["ret"].rolling(20, min_periods=12).std()
    cand[s] = pd.Series(safe_div(v5, v20), index=df.index)
report("vol_ratio_5x20", cand)

# 4-7. macro-linkage correlation: 60d rolling corr of asset ret with benchmark ret
def corr_with(bench):
    brets = series[bench]["ret"]
    out = {}
    for s, df in series.items():
        joined = pd.concat([df["ret"], brets], axis=1, join="outer")
        joined.columns = ["a", "b"]
        c = joined["a"].rolling(60, min_periods=40).corr(joined["b"])
        out[s] = c.reindex(df.index)
    return out

report("us10y_corr_60", corr_with("US10Y"))
report("xau_corr_60", corr_with("XAU"))
report("wti_corr_60", corr_with("WTI"))
report("cn10y_corr_60", corr_with("CN10Y"))

# 8. skew_60: 60d skewness of returns (library has ret_skew_10 -> longer window)
cand = {}
for s, df in series.items():
    r = df["ret"]
    mu = roll_mean_s(r, 60, 40)
    sd = roll_std_s(r, 60, 40)
    m3 = roll_mean_s(r ** 3, 60, 40)
    cand[s] = pd.Series(safe_div(m3 - 3 * mu * (sd ** 2) - mu ** 3, sd ** 3), index=df.index)
report("skew_60", cand)

# 9. volume_ratio_20x60: volume trend
cand = {}
for s, df in series.items():
    v = df["volume"]
    v20 = roll_mean_s(v, 20, 10)
    v60 = roll_mean_s(v, 60, 30)
    cand[s] = pd.Series(safe_div(v20, v60), index=df.index)
report("volume_ratio_20x60", cand)

# 10. amihud_20: illiquidity proxy mean(|ret|/volume, 20)
cand = {}
for s, df in series.items():
    il = (df["ret"].abs() / df["volume"].replace(0, np.nan))
    cand[s] = roll_mean_s(il, 20, 10)
report("amihud_20", cand)

# 11. max_gain_20: max positive daily return over 20d (upside tail)
cand = {s: pd.Series(df["ret"].rolling(20, min_periods=10).max(), index=df.index) for s, df in series.items()}
report("max_gain_20", cand)

# 12. hl_rank_20: where close sits in the 20d high-low range
cand = {}
for s, df in series.items():
    hi = df["high"].rolling(20, min_periods=10).max()
    lo = df["low"].rolling(20, min_periods=10).min()
    cand[s] = pd.Series(safe_div(df["close"] - lo, hi - lo), index=df.index)
report("hl_rank_20", cand)

json.dump(results, open("scripts/miner_3_20260910_batchB_results.json", "w"), indent=1)
print("SAVED batchB results")
