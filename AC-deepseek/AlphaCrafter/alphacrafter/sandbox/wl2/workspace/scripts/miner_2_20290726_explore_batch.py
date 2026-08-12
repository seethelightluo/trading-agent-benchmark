"""miner_2 2029-07-26 batch exploration: new candidate factors for current regime.
Data visible through 2029-07-25. Master grid from 2020-01-01.
Candidates focus on reversal, vol-regime, trend-consistency, liquidity, cross-asset dispersion
(most library momentum factors decayed per 2029-05-31 revalidation).
Gates: |IC|>=0.0070, |ICIR|>=0.0840 at horizon 10.
"""
import sys, json
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

OUT = "scripts/miner_2_20290726_explore_batch_results.json"


def asset_series_full(days=3000):
    out = {}
    for s in ASSETS:
        df = load_asset(s, days=days)
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
print("assets with data:", len(series), sorted(series.keys()))
print("grid size:", len(GRID), "first:", GRID[0], "last:", GRID[-1])
fwd10 = to_grid({s: df["fwd10"] for s, df in series.items()})
fwd_by_h = fwd_by_horizon_dict(series, horizons=(1, 2, 3, 5, 10, 20))
dates = np.array(GRID)

spx = series["SPX"]["close"]
dxy = load_macro("DXY")
usdjpy = load_macro("USDJPY")
vix = load_macro("VIX")

factors = {}

# 1) rev_vol_10: short-term reversal scaled by vol: -ret10 / std20 (positive IC => reversal works)
for s, df in series.items():
    r10 = df["close"] / df["close"].shift(10) - 1.0
    v20 = df["ret"].rolling(20, min_periods=10).std()
    factors.setdefault("rev_vol_10", {})[s] = -r10 / v20.replace(0, np.nan)

# 2) vol_rank_252x60: percentile of 60d vol within trailing 252d (vol regime position)
for s, df in series.items():
    v60 = df["ret"].rolling(60, min_periods=40).std()
    rank = v60.rolling(252, min_periods=120).apply(
        lambda x: (x.iloc[-1] >= x).mean(), raw=False)
    factors.setdefault("vol_rank_252x60", {})[s] = rank

# 3) weekly_win_12: fraction of positive weeks over last 12 weeks (trend consistency)
for s, df in series.items():
    _idx = pd.to_datetime(df.index)
    _s = pd.Series(df["close"].values, index=_idx)
    wk = _s.resample("W-FRI").last().pct_change()
    wk.index = pd.to_datetime(wk.index)
    f = wk.rolling(12, min_periods=8).apply(lambda x: (x > 0).mean(), raw=False)
    factors.setdefault("weekly_win_12", {})[s] = f.reindex(df.index).ffill()

# 4) skew_252: daily return skewness over 252d (crash-risk / lottery)
for s, df in series.items():
    factors.setdefault("skew_252", {})[s] = df["ret"].rolling(252, min_periods=120).skew()

# 5) range_ratio_20x120: mean daily range / close over 20d vs 120d baseline (activity expansion)
for s, df in series.items():
    rng = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    r20 = rng.rolling(20, min_periods=10).mean()
    r120 = rng.rolling(120, min_periods=60).mean()
    factors.setdefault("range_ratio_20x120", {})[s] = r20 / r120.replace(0, np.nan)

# 6) xcorr_rank_60: mean pairwise correlation of 60d returns with all other assets (dispersion/crowding)
ret_mat = to_grid({s: df["ret"] for s, df in series.items()})
T, n = ret_mat.shape
xcorr = np.full((T, n), np.nan)
for t in range(60, T):
    seg = ret_mat[t - 59:t + 1]
    with np.errstate(invalid="ignore"):
        c = np.corrcoef(seg, rowvar=False)
    for j in range(n):
        row = c[j]
        other = np.delete(row, j)
        ok = np.isfinite(other)
        if ok.sum() >= 5:
            xcorr[t, j] = other[ok].mean()
factors["xcorr_rank_60"] = {s: pd.Series(xcorr[:, j], index=GRID) for j, s in enumerate(ASSETS)}

# 7) us10y_beta_60: 60d beta to US10Y daily change (rate sensitivity)
us10y = series["US10Y"]["close"] if "US10Y" in series else None
if us10y is not None:
    ychg = us10y.diff()
    for s, df in series.items():
        beta = df["ret"].rolling(60, min_periods=30).cov(ychg) / ychg.rolling(60, min_periods=30).var()
        factors.setdefault("us10y_beta_60", {})[s] = beta

# 8) hi_lo_60: close position within 60d high-low range (trend freshness at medium horizon)
for s, df in series.items():
    hi = df["close"].rolling(60, min_periods=20).max()
    lo = df["close"].rolling(60, min_periods=20).min()
    factors.setdefault("hi_lo_60", {})[s] = (df["close"] - lo) / (hi - lo).replace(0, np.nan)

# 9) eff_ratio_60: efficiency ratio (net move / gross path) over 60d
for s, df in series.items():
    net = (df["close"] - df["close"].shift(60)).abs()
    gross = df["ret"].abs().rolling(60, min_periods=30).sum()
    factors.setdefault("eff_ratio_60", {})[s] = net / gross.replace(0, np.nan)

# 10) amihud_20: illiquidity |ret|/volume averaged 20d (liquidity premium)
for s, df in series.items():
    ill = df["ret"].abs() / df["volume"].replace(0, np.nan)
    factors.setdefault("amihud_20", {})[s] = ill.rolling(20, min_periods=10).mean()

# 11) mom60_vol60: 60d momentum scaled by 60d vol (longer risk-adjusted momentum)
for s, df in series.items():
    r60 = df["close"] / df["close"].shift(60) - 1.0
    v60 = df["ret"].rolling(60, min_periods=40).std()
    factors.setdefault("mom60_vol60", {})[s] = r60 / v60.replace(0, np.nan)

# 12) vix_ratio_20x120: VIX 20d mean vs 120d mean (macro stress momentum)
if vix is not None:
    v20 = vix.rolling(20, min_periods=10).mean()
    v120 = vix.rolling(120, min_periods=60).mean()
    vratio = (v20 / v120.replace(0, np.nan)).reindex(GRID)
    # apply to all assets as macro-regime factor
    for s in ASSETS:
        factors.setdefault("vix_ratio_20x120", {})[s] = vratio

results = {}
print("\n=== CANDIDATE RESULTS (horizon 10) ===")
for name, fd in sorted(factors.items()):
    mat = to_grid(fd)
    cov, dge8 = coverage_stats(mat)
    ics = spearman_ic_matrix(mat, fwd10)
    if len(ics) < 200:
        print(f"{name}: SKIP (only {len(ics)} IC dates)")
        continue
    idx = np.array([t for t, _ in ics])
    icv = np.array([v for _, v in ics])
    summ = summarize(ics, dates, name, HORIZON)
    turn = turnover_10d_rank(cross_sectional_rank(mat))
    decay = decay_curve(mat, fwd_by_h)
    rho_map, rho_name, rho_max = library_pairwise_corr(mat)
    ok = abs(summ["ic"]) >= 0.0070 and abs(summ["icir"]) >= 0.0840
    rec = summ["regime"].get("last250", {})
    res = {
        "label": name, "horizon": HORIZON, "n_ic_dates": summ["n_ic_dates"],
        "ic": summ["ic"], "icir": summ["icir"], "hit": summ["hit"],
        "regime": summ["regime"], "coverage": round(cov, 4),
        "dates_ge8_frac": round(dge8, 4), "turnover_10d_rank": round(turn, 4),
        "decay": decay, "max_abs_library_correlation": rho_max,
        "max_corr_with": rho_name, "ok": ok,
    }
    results[name] = res
    print(f"{name:24s} IC={summ['ic']:+.4f} ICIR={summ['icir']:+.4f} hit={summ['hit']:.3f} "
          f"cov={cov:.3f} turn={turn:.3f} rho={rho_max:.3f} ok={ok} "
          f"| last250 IC={rec.get('ic', float('nan')):+.4f} ICIR={rec.get('icir', float('nan')):+.4f}")

json.dump(results, open(OUT, "w"), indent=1)
print("\nsaved:", OUT)
