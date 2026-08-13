"""miner_3 2032-01-22 screening of NOVEL factor candidates (not in library).
Ideas tested (all new vs factors/ library):
 1. rel_mom20_med      : 20d return minus cross-sectional median (idiosyncratic momentum)
 2. vol_trend_20x60    : SMA(volume,20)/SMA(volume,60)-1  (volume never used in library)
 3. overnight_gap_20   : mean(open/prev_close-1, 20d)     (library only uses close/open intraday drift)
 4. varratio_5x1_60    : var(5d ret)/(5*var(1d ret)) over 60d -> trend persistence (Hurst-like)
 5. autocorr_20        : rolling 20d autocorr of 1d returns (reversal/trend persistence)
 6. skew_60            : rolling 60d skewness of daily returns
 7. mom_consistency_20 : fraction of up days over 20d (count-based, vs gain_loss magnitude-based)
 8. copper_beta_cond_60x20 : beta to COPPER * COPPER 20d move (commodity channel)
 9. wti_beta_cond_60x20    : beta to WTI * WTI 20d move
10. range_eff_60       : (close-min60)/(max60-min60) position in 60d range
11. shadow_ratio_20    : mean((high-close)/(close-low), 20d) intraday shape
12. amihud_20          : mean(|ret|/volume, 20d) illiquidity proxy
Gates: |IC|>=0.0070, |ICIR|>=0.0840 (Spearman vs fwd10, daily cross-section, >=8 valid).
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_3_20260813_lib import (
    ASSETS, load_asset, to_grid, load_macro, cross_sectional_rank,
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
    d["prev_close"] = close.shift(1)
    series[s] = d
print("assets with data:", sorted(series.keys()))

fwd10 = to_grid({s: df["close"].shift(-HORIZON) / df["close"] - 1.0 for s, df in series.items()})
fwd_by_h = fwd_by_horizon_dict(series, horizons=(1, 2, 3, 5, 10, 20))

copper = series.get("COPPER")
wti = series.get("WTI")

candidates = {}

# 1 rel_mom20_med
mat20 = to_grid({s: df["close"] / df["close"].shift(20) - 1.0 for s, df in series.items()})
med = np.nanmedian(mat20, axis=1, keepdims=True)
candidates["rel_mom20_med"] = mat20 - med

# 2 vol_trend_20x60
panel = {}
for s, df in series.items():
    v20 = df["volume"].rolling(20, min_periods=5).mean()
    v60 = df["volume"].rolling(60, min_periods=15).mean()
    panel[s] = v20 / v60 - 1.0
candidates["vol_trend_20x60"] = to_grid(panel)

# 3 overnight_gap_20
panel = {}
for s, df in series.items():
    gap = df["open"] / df["prev_close"] - 1.0
    panel[s] = gap.rolling(20, min_periods=10).mean()
candidates["overnight_gap_20"] = to_grid(panel)

# 4 varratio_5x1_60
panel = {}
for s, df in series.items():
    r5 = df["close"].pct_change(5)
    r1 = df["ret"]
    v5 = r5.rolling(60, min_periods=30).var()
    v1 = r1.rolling(60, min_periods=30).var()
    panel[s] = v5 / (5.0 * v1)
candidates["varratio_5x1_60"] = to_grid(panel)

# 5 autocorr_20
panel = {}
for s, df in series.items():
    panel[s] = df["ret"].rolling(20, min_periods=10).corr(df["ret"].shift(1))
candidates["autocorr_20"] = to_grid(panel)

# 6 skew_60
panel = {}
for s, df in series.items():
    panel[s] = df["ret"].rolling(60, min_periods=30).skew()
candidates["skew_60"] = to_grid(panel)

# 7 mom_consistency_20
panel = {}
for s, df in series.items():
    pos = (df["ret"] > 0).astype(float)
    panel[s] = pos.rolling(20, min_periods=10).mean()
candidates["mom_consistency_20"] = to_grid(panel)

# 8 copper_beta_cond_60x20
if copper is not None:
    panel = {}
    cr = copper["ret"]
    cm = copper["close"] / copper["close"].shift(20) - 1.0
    for s, df in series.items():
        beta = roll_beta_cond(df["ret"], cr, 60, 30)
        panel[s] = beta * cm.reindex(df.index)
    candidates["copper_beta_cond_60x20"] = to_grid(panel)

# 9 wti_beta_cond_60x20
if wti is not None:
    panel = {}
    wr = wti["ret"]
    wm = wti["close"] / wti["close"].shift(20) - 1.0
    for s, df in series.items():
        beta = roll_beta_cond(df["ret"], wr, 60, 30)
        panel[s] = beta * wm.reindex(df.index)
    candidates["wti_beta_cond_60x20"] = to_grid(panel)

# 10 range_eff_60
panel = {}
for s, df in series.items():
    lo = df["close"].rolling(60, min_periods=30).min()
    hi = df["close"].rolling(60, min_periods=30).max()
    panel[s] = (df["close"] - lo) / (hi - lo).replace(0, np.nan)
candidates["range_eff_60"] = to_grid(panel)

# 11 shadow_ratio_20
panel = {}
for s, df in series.items():
    up = df["high"] - df["close"]
    dn = df["close"] - df["low"]
    ratio = up / dn.replace(0, np.nan)
    panel[s] = ratio.rolling(20, min_periods=10).mean()
candidates["shadow_ratio_20"] = to_grid(panel)

# 12 amihud_20
panel = {}
for s, df in series.items():
    il = df["ret"].abs() / df["volume"].replace(0, np.nan)
    panel[s] = il.rolling(20, min_periods=10).mean()
candidates["amihud_20"] = to_grid(panel)

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

json.dump(results, open("scripts/miner_3_20320122_screen_novel.json", "w"), indent=1, default=str)
print("DONE saved scripts/miner_3_20320122_screen_novel.json")
