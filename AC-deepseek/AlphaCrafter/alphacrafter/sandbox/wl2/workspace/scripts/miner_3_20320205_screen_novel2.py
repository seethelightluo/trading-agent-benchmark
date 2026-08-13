"""miner_3 2032-02-05 screening of NOVEL factor candidates batch 2 (not in library).
Motivation: batch 1 (2032-01-22) produced no persistable factor - only mom_consistency_20
passed IC/ICIR but rho=1.000 vs downside_freq_20 (evicted). This batch tries
differentiated concepts avoiding the momentum/range/beta families already covered:
 1. xau_corr_60       : 60d rolling corr of asset returns with XAU (safe-haven channel)
 2. r2_trend_60       : R^2 of log-price linear trend fit over 60d (trend quality)
 3. mom_curve_20x60   : 20d ret - 60d ret/3 (short vs medium momentum slope, no vol scale)
 4. kurt_60           : rolling 60d excess kurtosis of daily returns (tail shape)
 5. max_dd_60         : 1 - close/rolling_max(close,60) (drawdown depth, continuous)
 6. cn_beta_cond_60x20: beta to 000300.SH * 000300.SH 20d move (China channel)
 7. us10y_beta_cond_60x20: beta to US10Y * US10Y 20d move (rates channel)
 8. range_norm_20     : mean((high-low)/close, 20d) amplitude (vol proxy family diff)
 9. hi_lo_pos_20      : mean((close-low)/(high-low), 20d) intraday close location
10. ret3d_rev         : -3d return (short-horizon reversal probe)
11. streak_share_20x5 : 5d ret / 20d ret (share of momentum earned recently; acceleration)
12. drawdown_252      : (close - max252)/max252 distance from 252d high (continuous)
Gates: |IC|>=0.0070, |ICIR|>=0.0840 (Spearman vs fwd10, daily cross-section, >=8 valid).
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_3_20260813_lib import (
    ASSETS, load_asset, to_grid, cross_sectional_rank,
    spearman_ic_matrix, summarize, decay_curve, fwd_by_horizon_dict,
    turnover_10d_rank, library_pairwise_corr, coverage_stats,
    HORIZON, MIN_ASSETS, GRID,
)

DAYS = 3600
dates = np.array(GRID)


def roll_beta_cond(asset_ret, ref_ret, w, minp):
    out = pd.Series(np.nan, index=asset_ret.index)
    a = asset_ret.values.astype(float)
    b = ref_ret.reindex(asset_ret.index).values.astype(float)
    for i in range(w - 1, len(a)):
        seg = slice(i - w + 1, i + 1)
        x = b[seg]; y = a[seg]
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() < minp:
            continue
        xv = x[m]; yv = y[m]
        if np.std(xv) < 1e-12:
            continue
        beta = np.cov(xv, yv)[0, 1] / np.var(xv)
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
    d = pd.DataFrame({
        "close": close, "ret": close.pct_change(),
        "open": df["open"].astype(float), "high": df["high"].astype(float),
        "low": df["low"].astype(float), "volume": df["volume"].astype(float),
    })
    series[s] = d
print("assets with data:", sorted(series.keys()))

fwd10 = to_grid({s: df["close"].shift(-HORIZON) / df["close"] - 1.0 for s, df in series.items()})
fwd_by_h = fwd_by_horizon_dict(series, horizons=(1, 2, 3, 5, 10, 20))

xau = series.get("XAU")
cn = series.get("000300.SH")
us10y = series.get("US10Y")

candidates = {}

# 1 xau_corr_60
if xau is not None:
    panel = {}
    xr = xau["ret"]
    for s, df in series.items():
        panel[s] = df["ret"].rolling(60, min_periods=30).corr(xr.reindex(df.index))
    candidates["xau_corr_60"] = to_grid(panel)

# 2 r2_trend_60
panel = {}
for s, df in series.items():
    lp = np.log(df["close"].values.astype(float))
    n = len(lp)
    out = np.full(n, np.nan)
    t = np.arange(60, dtype=float)
    for i in range(59, n):
        seg = lp[i - 59:i + 1]
        if not np.all(np.isfinite(seg)):
            continue
        if np.std(seg) < 1e-12:
            continue
        c = np.corrcoef(t, seg)[0, 1]
        out[i] = c * c
    panel[s] = pd.Series(out, index=df.index)
candidates["r2_trend_60"] = to_grid(panel)

# 3 mom_curve_20x60
panel = {}
for s, df in series.items():
    r20 = df["close"] / df["close"].shift(20) - 1.0
    r60 = df["close"] / df["close"].shift(60) - 1.0
    panel[s] = r20 - r60 / 3.0
candidates["mom_curve_20x60"] = to_grid(panel)

# 4 kurt_60
panel = {}
for s, df in series.items():
    panel[s] = df["ret"].rolling(60, min_periods=30).kurt()
candidates["kurt_60"] = to_grid(panel)

# 5 max_dd_60
panel = {}
for s, df in series.items():
    hi = df["close"].rolling(60, min_periods=30).max()
    panel[s] = 1.0 - df["close"] / hi
candidates["max_dd_60"] = to_grid(panel)

# 6 cn_beta_cond_60x20
if cn is not None:
    panel = {}
    cr = cn["ret"]
    cm = cn["close"] / cn["close"].shift(20) - 1.0
    for s, df in series.items():
        beta = roll_beta_cond(df["ret"], cr, 60, 30)
        panel[s] = beta * cm.reindex(df.index)
    candidates["cn_beta_cond_60x20"] = to_grid(panel)

# 7 us10y_beta_cond_60x20
if us10y is not None:
    panel = {}
    ur = us10y["ret"]
    um = us10y["close"] / us10y["close"].shift(20) - 1.0
    for s, df in series.items():
        beta = roll_beta_cond(df["ret"], ur, 60, 30)
        panel[s] = beta * um.reindex(df.index)
    candidates["us10y_beta_cond_60x20"] = to_grid(panel)

# 8 range_norm_20
panel = {}
for s, df in series.items():
    amp = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    panel[s] = amp.rolling(20, min_periods=10).mean()
candidates["range_norm_20"] = to_grid(panel)

# 9 hi_lo_pos_20
panel = {}
for s, df in series.items():
    rng = df["high"] - df["low"]
    pos = (df["close"] - df["low"]) / rng.replace(0, np.nan)
    panel[s] = pos.rolling(20, min_periods=10).mean()
candidates["hi_lo_pos_20"] = to_grid(panel)

# 10 ret3d_rev
panel = {}
for s, df in series.items():
    r3 = df["close"] / df["close"].shift(3) - 1.0
    panel[s] = -r3
candidates["ret3d_rev"] = to_grid(panel)

# 11 streak_share_20x5
panel = {}
for s, df in series.items():
    r20 = df["close"] / df["close"].shift(20) - 1.0
    r5 = df["close"] / df["close"].shift(5) - 1.0
    panel[s] = r5 / r20.replace(0, np.nan)
candidates["streak_share_20x5"] = to_grid(panel)

# 12 drawdown_252
panel = {}
for s, df in series.items():
    hi = df["close"].rolling(252, min_periods=120).max()
    panel[s] = (df["close"] - hi) / hi
candidates["drawdown_252"] = to_grid(panel)

results = {}
for fid, mat in candidates.items():
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(rank_mat, fwd10)
    if len(ics) == 0:
        print(fid, "NO IC DATES")
        continue
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
    reg = s["regime"].get("last250", {})
    print(f"{fid:26s} ic={s['ic']:+.4f} icir={s['icir']:+.4f} hit={s['hit']:.3f} "
          f"turn={s['turnover_10d_rank']:.3f} cov={s['coverage']:.3f} maxrho={max_rho:.3f} "
          f"({rho_name}) l250={reg.get('ic','NA')} ok={s['ok']}", flush=True)

json.dump(results, open("scripts/miner_3_20320205_screen_novel2.json", "w"), indent=1, default=str)
print("DONE saved scripts/miner_3_20320205_screen_novel2.json")
