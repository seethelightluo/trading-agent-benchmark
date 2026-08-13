"""miner_1 2031-07-24: exploration batch - mean-reversion / contrarian / regime-aware candidates.
Motivation: baseline revalidation (2031-07-24) shows short & long momentum flipping sign and
range_pos_252 strongly negative (IC -0.0341, last250 -0.144) => current tape is mean-reverting.
Candidates below are per-asset own-calendar, reindexed to master grid, IC vs fwd10d.
All macro series truncated to visible date 2031-07-23 (CSVs contain future rows).
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_3_20260813_lib import (
    GRID, ASSETS, load_asset, load_macro, to_grid,
    cross_sectional_rank, spearman_ic_matrix,
    summarize, decay_curve, fwd_by_horizon_dict, turnover_10d_rank,
    library_pairwise_corr, coverage_stats, HORIZON,
)

VISIBLE = "2031-07-23"
DAYS = 3400


def load_macro_visible(name):
    s = load_macro(name)
    return None if s is None else s[s.index <= VISIBLE]


def asset_series_full():
    out = {}
    for s in ASSETS:
        df = load_asset(s, days=DAYS)
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
print("assets:", len(series))
fwd10 = to_grid({s: df["fwd10"] for s, df in series.items()})
fwd_by_h = fwd_by_horizon_dict(series, horizons=(1, 2, 3, 5, 10, 20))
dates = np.array(GRID)

spx = series["SPX"]["close"]
dxy = load_macro_visible("DXY")
vix = load_macro_visible("VIX")
usdjpy = load_macro_visible("USDJPY")

factors = {}

# 1. reversal_5d: -5d momentum (short-horizon contrarian)
for s, df in series.items():
    factors.setdefault("reversal_5d", {})[s] = -(df["close"] / df["close"].shift(5) - 1.0)

# 2. reversal_5d_volscale: -5d mom / 20d vol (vol-scaled contrarian)
for s, df in series.items():
    m = df["close"] / df["close"].shift(5) - 1.0
    v = df["ret"].rolling(20, min_periods=10).std()
    factors.setdefault("reversal_5d_volscale", {})[s] = -m / v.replace(0, np.nan)

# 3. bollinger_z_20: (close - sma20)/std20 (overbought -> negative fwd in MR regime)
for s, df in series.items():
    ma = df["close"].rolling(20, min_periods=10).mean()
    sd = df["close"].rolling(20, min_periods=10).std()
    factors.setdefault("bollinger_z_20", {})[s] = (df["close"] - ma) / sd.replace(0, np.nan)

# 4. rsi_14: classic RSI (overbought high -> contrarian negative)
for s, df in series.items():
    r = df["ret"]
    up = r.clip(lower=0).rolling(14, min_periods=7).mean()
    dn = (-r).clip(lower=0).rolling(14, min_periods=7).mean()
    rs = up / dn.replace(0, np.nan)
    factors.setdefault("rsi_14", {})[s] = 100 - 100 / (1 + rs)

# 5. drawup_20: 20d run-up above rolling min (recent climb -> negative)
for s, df in series.items():
    roll_min = df["close"].rolling(60, min_periods=20).min()
    factors.setdefault("drawup_20", {})[s] = -(df["close"] / roll_min - 1.0)

# 6. eff_ratio_20_rev: -efficiency ratio (choppy -> positive, trending -> negative)
for s, df in series.items():
    num = (df["close"] - df["close"].shift(20)).abs()
    den = df["ret"].abs().rolling(20, min_periods=10).sum()
    factors.setdefault("eff_ratio_20_rev", {})[s] = -num / den.replace(0, np.nan)

# 7. skew_20: rolling return skewness (negative skew -> bounce?)
for s, df in series.items():
    factors.setdefault("ret_skew_20", {})[s] = df["ret"].rolling(20, min_periods=10).skew()

# 8. vol_regime_z: VIX z-score (high fear -> mean-revert up); cross-sectional via beta*level
vix_z = (vix - vix.rolling(60, min_periods=30).mean()) / vix.rolling(60, min_periods=30).std()
for s, df in series.items():
    b = df["ret"].rolling(60, min_periods=30).corr(-vix.reindex(df.index).pct_change())
    factors.setdefault("vix_z_beta_60", {})[s] = b * vix_z.reindex(df.index)

# 9. yield_level_carry: US10Y/CN10Y level vs 252d MA (yield carry); for others use cross-mkt mom
for s, df in series.items():
    if s in ("US10Y", "CN10Y"):
        ma = df["close"].rolling(252, min_periods=120).mean()
        factors.setdefault("yield_carry_252", {})[s] = -(df["close"] / ma - 1.0)
    else:
        factors.setdefault("yield_carry_252", {})[s] = pd.Series(np.nan, index=df.index)

# 10. range_squeeze_20: (high-low)/close rolling 20d mean (squeeze -> expansion)
for s, df in series.items():
    rng = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    factors.setdefault("range_squeeze_20", {})[s] = -rng.rolling(20, min_periods=10).mean()

# 11. crypto_spill_20: 20d BTC momentum as cross-asset predictor (risk appetite)
btc_ret20 = (series["BTC"]["close"] / series["BTC"]["close"].shift(20) - 1.0)
for s, df in series.items():
    factors.setdefault("crypto_spill_20", {})[s] = btc_ret20.reindex(df.index)

# 12. downfreq_20_rev: -frequency of down days over 20d (contrarian bounce)
for s, df in series.items():
    dn = (df["ret"] < 0).astype(float).rolling(20, min_periods=10).mean()
    factors.setdefault("downfreq_20_rev", {})[s] = -dn

results = {}
for fid, panel in factors.items():
    mat = to_grid(panel)
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(rank_mat, fwd10)
    if not ics:
        print(fid, "NO IC DATES")
        continue
    s = summarize(ics, dates, fid, HORIZON)
    rho_dict, rho_name, max_rho = library_pairwise_corr(mat)
    s["max_abs_library_correlation"] = round(max_rho, 4)
    s["max_corr_with"] = rho_name
    s["turnover_10d_rank"] = round(turnover_10d_rank(rank_mat), 4)
    cov, dates_ge8 = coverage_stats(mat)
    s["coverage"] = round(cov, 4)
    s["dates_ge8_frac"] = round(dates_ge8, 4)
    s["decay"] = decay_curve(rank_mat, fwd_by_h)
    s["ok"] = bool((abs(s["ic"]) >= 0.0070) and (abs(s["icir"]) >= 0.0840))
    icv = np.array([v for _, v in ics])
    idx = np.array([t for t, _ in ics])
    if len(icv) >= 250:
        m = idx >= len(dates) - 250
        s["last250_ic"] = round(float(np.mean(icv[m])), 4)
        s["last250_icir"] = round(float(np.mean(icv[m]) / np.std(icv[m])), 3) if np.std(icv[m]) > 0 else 0.0
    if len(icv) >= 750:
        m = idx >= len(dates) - 750
        s["last750_ic"] = round(float(np.mean(icv[m])), 4)
        s["last750_icir"] = round(float(np.mean(icv[m]) / np.std(icv[m])), 3) if np.std(icv[m]) > 0 else 0.0
    s.pop("idx", None); s.pop("icv", None)
    results[fid] = s
    print(f"{fid:24s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['hit']:.3f} n={s['n_ic_dates']} "
          f"cov={s['coverage']:.3f} turn={s['turnover_10d_rank']:.3f} maxrho={max_rho:.3f} "
          f"l250={s.get('last250_ic','NA')}/{s.get('last250_icir','NA')} l750={s.get('last750_ic','NA')} "
          f"decay={s['decay']} -> {'PASS' if s['ok'] else 'fail'}")

json.dump(results, open("scripts/miner_1_20310724_explore_batchA.json", "w"), indent=1, default=str)
print("\nSaved scripts/miner_1_20310724_explore_batchA.json")
