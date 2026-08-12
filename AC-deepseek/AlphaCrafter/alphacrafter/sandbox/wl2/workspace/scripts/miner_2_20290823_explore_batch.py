"""miner_2 2029-08-23: batch exploration of novel factor candidates on the 15-instrument
cross-asset universe. Data visible through 2029-08-22 (current sim date 2029-08-23).
Conventions match miner_3 shared lib: IC = daily cross-sectional Spearman vs own-calendar
fwd10 return; gates |IC|>=0.0070 and |ICIR|>=0.0840.
This script only EXPLORES; persistence happens after focused validation.
"""
import sys, json, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_3_20260813_lib import (
    GRID, ASSETS, load_asset, to_grid, load_macro, safe_div,
    cross_sectional_rank, spearman_ic_matrix, summarize, decay_curve,
    fwd_by_horizon_dict, turnover_10d_rank, library_pairwise_corr, coverage_stats,
    HORIZON, MIN_ASSETS,
)

DAYS = 3400
series = {}
for s in ASSETS:
    df = load_asset(s, days=DAYS)
    if df is None or len(df) < 200:
        print("SKIP asset", s)
        continue
    close = df["close"].astype(float)
    ret = close.pct_change()
    d = pd.DataFrame({
        "close": close, "ret": ret,
        "open": df["open"].astype(float), "high": df["high"].astype(float),
        "low": df["low"].astype(float), "volume": df["volume"].astype(float),
    })
    d["prev_close"] = close.shift(1)
    d["gap"] = d["open"] / d["prev_close"] - 1.0
    d["intraday"] = d["close"] / d["open"] - 1.0
    series[s] = d

print("assets:", len(series), "grid rows:", len(GRID), "grid last:", GRID[-1])

spx = series["SPX"]["close"]
dxy = load_macro("DXY")
vix = load_macro("VIX")
us10y = series["US10Y"]["close"]

fwd_by_h = fwd_by_horizon_dict(series, horizons=(1, 2, 3, 5, 10, 20))
dates = np.array(GRID)


def roll_beta(asset_ret, ref_ret, w, minp):
    out = pd.Series(np.nan, index=asset_ret.index)
    a = asset_ret.values.astype(float)
    b = ref_ret.reindex(asset_ret.index).values.astype(float)
    for i in range(w - 1, len(a)):
        seg = slice(i - w + 1, i + 1)
        x = b[seg]; y = a[seg]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < minp or np.std(x[ok]) < 1e-12:
            continue
        beta = np.cov(x[ok], y[ok])[0, 1] / np.var(x[ok])
        if np.isfinite(beta):
            out.iloc[i] = beta
    return out


def roll_corr(a_ret, b_ret, w, minp):
    out = pd.Series(np.nan, index=a_ret.index)
    a = a_ret.values.astype(float)
    b = b_ret.reindex(a_ret.index).values.astype(float)
    for i in range(w - 1, len(a)):
        seg = slice(i - w + 1, i + 1)
        x = a[seg]; y = b[seg]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < minp:
            continue
        c = np.corrcoef(x[ok], y[ok])[0, 1]
        if np.isfinite(c):
            out.iloc[i] = c
    return out


def roll_skew(x, w, minp):
    return x.rolling(w, min_periods=minp).skew()


def roll_kurt(x, w, minp):
    return x.rolling(w, min_periods=minp).kurt()


def r2_trend(close, w, minp):
    out = pd.Series(np.nan, index=close.index)
    y = np.log(close.values.astype(float))
    t = np.arange(w, dtype=float)
    for i in range(w - 1, len(y)):
        seg = slice(i - w + 1, i + 1)
        yy = y[seg]
        if np.isfinite(yy).sum() < minp:
            continue
        xx = t[np.isfinite(yy)]
        yy = yy[np.isfinite(yy)]
        if len(xx) < minp or np.std(xx) < 1e-12:
            continue
        r = np.corrcoef(xx, yy)[0, 1]
        if np.isfinite(r):
            out.iloc[i] = r * r
    return out


factors = {}

# 1 skew_60: return skewness
for s, df in series.items():
    factors.setdefault("skew_60", {})[s] = roll_skew(df["ret"], 60, 30)

# 2 kurt_60: excess kurtosis
for s, df in series.items():
    factors.setdefault("kurt_60", {})[s] = roll_kurt(df["ret"], 60, 30)

# 3 r2_trend_60: log-price trend fit R^2
for s, df in series.items():
    factors.setdefault("r2_trend_60", {})[s] = r2_trend(df["close"], 60, 30)

# 4 gap_ratio_20: overnight-fraction of total daily move
for s, df in series.items():
    g = df["gap"].abs()
    id_ = df["intraday"].abs()
    num = g.rolling(20, min_periods=10).mean()
    den = (g + id_).rolling(20, min_periods=10).mean()
    factors.setdefault("gap_ratio_20", {})[s] = safe_div(num, den.replace(0, np.nan))

# 5 volume_zscore_20: abnormal volume (attention)
for s, df in series.items():
    v = df["volume"].astype(float)
    mu = v.rolling(20, min_periods=10).mean()
    sd = v.rolling(20, min_periods=10).std()
    z = safe_div(v - mu, sd)
    factors.setdefault("volume_zscore_20", {})[s] = z.clip(-5, 5)

# 6 vix_beta_60: beta to VIX daily % change (unconditional risk-off sensitivity)
vix_ret = vix.pct_change() if vix is not None else None
for s, df in series.items():
    if vix_ret is None:
        continue
    factors.setdefault("vix_beta_60", {})[s] = roll_beta(df["ret"], vix_ret, 60, 30)

# 7 us10y_corr_60: rolling corr with US10Y yield daily change
us10y_chg = us10y.diff()
for s, df in series.items():
    factors.setdefault("us10y_corr_60", {})[s] = roll_corr(df["ret"], us10y_chg, 60, 30)

# 8 amihud_20: illiquidity |ret|/volume (log)
for s, df in series.items():
    v = df["volume"].astype(float).replace(0, np.nan)
    illiq = (df["ret"].abs() / v)
    factors.setdefault("amihud_20", {})[s] = np.log1p(illiq.rolling(20, min_periods=10).mean() * 1e6)

# 9 zscore_rev_20d: distance of price from 20d mean in std units (mean reversion)
for s, df in series.items():
    mu = df["close"].rolling(20, min_periods=10).mean()
    sd = df["close"].rolling(20, min_periods=10).std()
    factors.setdefault("zscore_rev_20d", {})[s] = safe_div(df["close"] - mu, sd)

print("\n=== BATCH EXPLORATION RESULTS (fwd10 Spearman IC, 2020-01-01..2029-08-22) ===")
results = {}
for name, fdict in factors.items():
    fmat = to_grid(fdict)
    ics = spearman_ic_matrix(fmat, fwd_by_h[10])
    if not ics:
        print(name, "NO IC DATES"); continue
    idx = np.array([t for t, _ in ics]); icv = np.array([v for _, v in ics])
    ic = float(np.nanmean(icv)); sd = float(np.nanstd(icv))
    icir = ic / sd if sd > 0 else 0.0
    hit = float(np.mean(icv > 0))
    cov_ad, cov_d8 = coverage_stats(fmat)
    turn = turnover_10d_rank(cross_sectional_rank(fmat))
    dec = decay_curve(fmat, fwd_by_h)
    corr_map, corr_name, corr_max = library_pairwise_corr(fmat)
    reg = {}
    segs = [("2020-21", "2020-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
            ("2023-24", "2023-01-01", "2024-12-31"), ("2025-26", "2025-01-01", "2026-12-31"),
            ("2027-29", "2027-01-01", "2099-12-31")]
    for rn, a, b in segs:
        m = (dates[idx] >= a) & (dates[idx] <= b)
        if m.sum() > 20:
            sdm = float(np.std(icv[m]))
            reg[rn] = [round(float(np.mean(icv[m])), 4), round(float(np.mean(icv[m]) / sdm), 3) if sdm > 0 else 0.0, int(m.sum())]
    if len(icv) >= 250:
        m = idx >= len(dates) - 250
        sdm = float(np.std(icv[m]))
        reg["last250"] = [round(float(np.mean(icv[m])), 4), round(float(np.mean(icv[m]) / sdm), 3) if sdm > 0 else 0.0, int(m.sum())]
    results[name] = {
        "ic": ic, "icir": icir, "hit": hit, "n": len(icv),
        "coverage_asset_days": cov_ad, "coverage_dates_ge8": cov_d8,
        "turnover_10d_rank": turn, "decay": dec,
        "max_abs_library_correlation": corr_max, "top_corr": corr_name,
        "regime": reg,
    }
    print(f"\n{name}: IC={ic:.4f} ICIR={icir:.4f} hit={hit:.3f} n={len(icv)} cov={cov_ad:.3f}/{cov_d8:.3f} turn={turn:.3f} maxlibcorr={corr_max:.3f}({corr_name})")
    print("  decay:", dec)
    print("  regime:", reg)

with open("scripts/miner_2_20290823_explore_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved explore results.")
