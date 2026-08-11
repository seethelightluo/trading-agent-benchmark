"""miner_3 2026-09-10 exploration batch C: oscillator/mean-reversion, drawdown,
momentum acceleration, beta trend/shift, EURUSD & USDCNY & US10Y macro betas,
short-window VIX beta, volume momentum, range-noise ratio, MA-gap.

Motivation: library covers long-horizon trend position (range_pos_252,
days_since_high_60, close_pos_20, hl_rank_20), skip-5 momentum (10/20/120/180),
vol level/cluster (volcluster_60, vol_of_vol20x60, updown_vol_ratio_20,
downside_freq_20), extremes (max_gain_20, max_consec_*), macro betas to
SPX/DXY/USDJPY/VIX(cond)/CN10Y. NOT covered: RSI, z-score distance, drawdown
depth, momentum acceleration (curvature), beta *trend* (shift), EURUSD/USDCNY
linkage (never used), unconditional short-window VIX beta, US10Y *beta* (only
corr was tried), volume momentum 5x60, range-noise ratio, MA-gap.

Gates: |IC|>=0.007, |ICIR|>=0.084 on daily cross-sectional Spearman vs fwd 10d.
Library correlation: report max |rho| vs factors/*.signal.npy (informational).
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
                                  coverage_stats, safe_div, load_macro)

GATE_IC = 0.0070
GATE_ICIR = 0.0840


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
    print("=" * 100)
    print(f"{name}: IC={ic:+.4f} ICIR={icir:+.4f} hit={summ['hit']:.3f} n={summ['n_ic_dates']} "
          f"cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} turn={to:.3f} maxlibcorr={mx_abs:.3f} ({mx_name}) "
          f"GATE={ok}")
    print("   regime:", {k: v for k, v in summ["regime"].items()})
    print("   decay:", dec)
    results[name] = {
        "ic": round(ic, 5), "icir": round(icir, 5), "hit": round(summ["hit"], 4),
        "n_ic_dates": summ["n_ic_dates"], "regime": summ["regime"],
        "coverage_asset_days": round(cov_ad, 4), "coverage_dates_ge8": round(cov_d8, 4),
        "turnover_10d_rank": round(to, 4), "decay": dec,
        "max_abs_library_correlation": round(mx_abs, 4),
        "max_lib_corr_name": mx_name, "pass_gate": bool(ok),
    }
    return summ


# 1. rsi_14: classic oscillator (simple-mean approximation)
cand = {}
for s, df in series.items():
    d = df["ret"].clip(lower=0.0)
    u = df["ret"].clip(upper=0.0).abs()
    ag = d.rolling(14, min_periods=8).mean()
    al = u.rolling(14, min_periods=8).mean()
    rs = safe_div(ag, al)
    cand[s] = pd.Series(100.0 - 100.0 / (1.0 + rs), index=df.index)
report("rsi_14", cand)

# 2. zscore_20: normalized distance of close from 20d mean
cand = {}
for s, df in series.items():
    ma = df["close"].rolling(20, min_periods=10).mean()
    sd = df["close"].rolling(20, min_periods=10).std()
    cand[s] = pd.Series(safe_div(df["close"] - ma, sd), index=df.index)
report("zscore_20", cand)

# 3. drawdown_252: current drawdown depth from trailing 252d peak
cand = {}
for s, df in series.items():
    pk = df["close"].rolling(252, min_periods=60).max()
    cand[s] = pd.Series(safe_div(df["close"] - pk, pk), index=df.index)
report("drawdown_252", cand)

# 4. mom_accel_20x60: momentum acceleration (20d mom minus 60d mom, skip 5)
cand = {}
for s, df in series.items():
    m20 = df["close"].pct_change(20).shift(5)
    m60 = df["close"].pct_change(60).shift(5)
    cand[s] = pd.Series(m20 - m60, index=df.index)
report("mom_accel_20x60", cand)

# 5. beta_shift_spx_20: change in 60d SPX beta over trailing 20d
spx_ret = series["SPX"]["ret"]
cand = {}
for s, df in series.items():
    joined = pd.concat([df["ret"], spx_ret], axis=1, join="outer")
    joined.columns = ["a", "b"]
    cov = joined["a"].rolling(60, min_periods=40).cov(joined["b"])
    var = joined["b"].rolling(60, min_periods=40).var()
    beta = pd.Series(safe_div(cov, var), index=joined.index)
    beta = beta.reindex(df.index)
    cand[s] = pd.Series(beta - beta.shift(20), index=df.index)
report("beta_shift_spx_20", cand)

# 6. eurusd_beta_60: 60d beta of asset ret vs EURUSD ret (macro, observation-only)
m = load_macro("EURUSD")
m_ret = m.pct_change()
cand = {}
for s, df in series.items():
    joined = pd.concat([df["ret"], m_ret], axis=1, join="outer")
    joined.columns = ["a", "b"]
    cov = joined["a"].rolling(60, min_periods=40).cov(joined["b"])
    var = joined["b"].rolling(60, min_periods=40).var()
    tmp = pd.Series(safe_div(cov, var), index=joined.index)
    cand[s] = tmp.reindex(df.index)
report("eurusd_beta_60", cand)

# 7. usdcny_beta_60: 60d beta of asset ret vs USDCNY ret
m = load_macro("USDCNY")
m_ret = m.pct_change()
cand = {}
for s, df in series.items():
    joined = pd.concat([df["ret"], m_ret], axis=1, join="outer")
    joined.columns = ["a", "b"]
    cov = joined["a"].rolling(60, min_periods=40).cov(joined["b"])
    var = joined["b"].rolling(60, min_periods=40).var()
    tmp = pd.Series(safe_div(cov, var), index=joined.index)
    cand[s] = tmp.reindex(df.index)
report("usdcny_beta_60", cand)

# 8. vix_beta_20: unconditional short-window beta vs VIX ret
m = load_macro("VIX")
m_ret = m.pct_change()
cand = {}
for s, df in series.items():
    joined = pd.concat([df["ret"], m_ret], axis=1, join="outer")
    joined.columns = ["a", "b"]
    cov = joined["a"].rolling(20, min_periods=12).cov(joined["b"])
    var = joined["b"].rolling(20, min_periods=12).var()
    tmp = pd.Series(safe_div(cov, var), index=joined.index)
    cand[s] = tmp.reindex(df.index)
report("vix_beta_20", cand)

# 9. us10y_beta_20: 20d beta vs US10Y ret (yield-sensitivity beta; corr variant failed)
m = load_macro.__self__ if False else None
us10 = series["US10Y"]["ret"]
cand = {}
for s, df in series.items():
    joined = pd.concat([df["ret"], us10], axis=1, join="outer")
    joined.columns = ["a", "b"]
    cov = joined["a"].rolling(20, min_periods=12).cov(joined["b"])
    var = joined["b"].rolling(20, min_periods=12).var()
    tmp = pd.Series(safe_div(cov, var), index=joined.index)
    cand[s] = tmp.reindex(df.index)
report("us10y_beta_20", cand)

# 10. volume_mom_5x60: short-term volume expansion vs long baseline
cand = {}
for s, df in series.items():
    v = df["volume"]
    v5 = v.rolling(5, min_periods=4).mean()
    v60 = v.rolling(60, min_periods=30).mean()
    cand[s] = pd.Series(safe_div(v5, v60), index=df.index)
report("volume_mom_5x60", cand)

# 11. range_vol_ratio_20: intraday range vol / close-to-close vol (noise ratio)
cand = {}
for s, df in series.items():
    vr = df["rng_pct"].rolling(20, min_periods=12).std()
    cr = df["ret"].rolling(20, min_periods=12).std()
    cand[s] = pd.Series(safe_div(vr, cr), index=df.index)
report("range_vol_ratio_20", cand)

# 12. ma_gap_10x60: (MA10 - MA60)/close trend steepness
cand = {}
for s, df in series.items():
    ma10 = df["close"].rolling(10, min_periods=6).mean()
    ma60 = df["close"].rolling(60, min_periods=30).mean()
    cand[s] = pd.Series(safe_div(ma10 - ma60, df["close"]), index=df.index)
report("ma_gap_10x60", cand)

json.dump(results, open("scripts/miner_3_20260910_batchC_results.json", "w"), indent=1)
print("SAVED batchC results")
