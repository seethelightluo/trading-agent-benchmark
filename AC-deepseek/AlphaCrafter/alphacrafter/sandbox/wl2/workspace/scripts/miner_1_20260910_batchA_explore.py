"""miner_1 2026-09-10 batch A exploration: trend-distance, serial dependence, drawdown depth,
macro-beta (global risk-on, yield spread, safe-haven, energy), cross-sectional relative strength,
liquidity/volume flow.

Motivation: existing library is rich in momentum (mom_*_skip5, mom20_volproxy60, mom30_vol60),
trend position (range_pos_252, days_since_high_60, close_pos_20), vol level/cluster
(volcluster_60, vol_of_vol20x60), and macro beta to SPX/DXY/VIX/USDJPY.  Missing: price-vs-mean
statistical distance, return serial dependence (autocorrelation / long memory), drawdown DEPTH
(not time-since-high), beta to equal-weight cross-asset risk index, beta to the US10Y-CN10Y
yield spread, safe-haven (XAU) beta, energy (WTI) beta, 40d relative strength vs cross-sectional
median, cross-sectional vol z-score, Amihud illiquidity, and volume flow ratio.

Gates: |IC|>=0.007, |ICIR|>=0.084 on daily cross-sectional Spearman vs fwd 10d (own calendar).
Library correlation target max |rho| < 0.5.
"""
import sys, json, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict
from miner_3_20260813_lib import (ASSETS, GRID, N_GRID, HORIZON, to_grid, cross_sectional_rank,
                                  spearman_ic_matrix, summarize, decay_curve, fwd_by_horizon_dict,
                                  turnover_10d_rank, library_pairwise_corr, coverage_stats, safe_div,
                                  roll_mean, roll_std, load_macro, load_asset)

series = {}
for s in ASSETS:
    df = load_asset(s)
    if df is not None and len(df) > 120:
        close = df["close"].astype(float)
        ret = close.pct_change()
        d = pd.DataFrame({"close": close, "ret": ret})
        d["logp"] = np.log(close)
        d["vol20"] = ret.rolling(20).std()
        d["vol60"] = ret.rolling(60).std()
        d["sma20"] = close.rolling(20).mean()
        d["sma40"] = close.rolling(40).mean()
        d["max60"] = close.rolling(60).max()
        d["vol5"] = ret.rolling(5).std()
        series[s] = d
print(f"assets loaded: {len(series)}/15 -> {sorted(series.keys())}")

# ---------- helpers ----------
def rolling_corr(x, y, w):
    """rolling Pearson corr between two aligned Series over window w."""
    mx = x.rolling(w).mean(); my = y.rolling(w).mean()
    cov = ((x - mx) * (y - my)).rolling(w).sum()
    vx = ((x - mx) ** 2).rolling(w).sum()
    vy = ((y - my) ** 2).rolling(w).sum()
    return cov / np.sqrt(vx * vy)

def rolling_beta(x, y, w):
    """beta of y on x (x = regressor) over window w."""
    mx = x.rolling(w).mean(); my = y.rolling(w).mean()
    cov = ((x - mx) * (y - my)).rolling(w).sum()
    vx = ((x - mx) ** 2).rolling(w).sum()
    return cov / vx

def hurst_rs(logp, w=60):
    """Rolling R/S Hurst exponent on log-price window w."""
    ret = logp.diff()
    out = pd.Series(np.nan, index=logp.index)
    for i in range(w, len(logp) + 1):
        seg = ret.iloc[i - w:i].values
        seg = seg[np.isfinite(seg)]
        if len(seg) < w * 0.7:
            continue
        mean = seg.mean()
        dev = np.cumsum(seg - mean)
        R = dev.max() - dev.min()
        S = seg.std(ddof=1)
        if S > 1e-12 and R > 0:
            out.iloc[i - 1] = np.log(R / S) / np.log(len(seg))
    return out

def rolling_skew(x, w):
    mu = x.rolling(w).mean()
    sd = x.rolling(w).std(ddof=0)
    m3 = ((x - mu) ** 3).rolling(w).mean()
    return m3 / (sd ** 3)

# ---------- cross-asset index returns on master grid ----------
ret_mat = to_grid({s: series[s]["ret"] for s in series})
ew_ret = np.nanmean(ret_mat, axis=1)
ew_ret_ser = pd.Series(ew_ret, index=GRID)
ew_log = np.log(1 + np.nan_to_num(ew_ret, nan=0.0)).cumsum()
print("EW index built, grid", len(GRID), "rows")

# macro
us10y = load_macro("US10Y") if os.path.exists("../persistent/index_data/US10Y.csv") else None
# NOTE: US10Y/CN10Y are tradable watchlist assets -> load from stock_data (own calendar)
yld = {}
for s in ["US10Y", "CN10Y"]:
    if s in series:
        yld[s] = series[s]["close"]
yspread = None
if "US10Y" in yld and "CN10Y" in yld:
    idx = yld["US10Y"].index.union(yld["CN10Y"].index).sort_values()
    yspread = (yld["US10Y"].reindex(idx) - yld["CN10Y"].reindex(idx)).dropna()
    yspread.name = "yspread"
print("yspread length:", None if yspread is None else len(yspread))

xau_ret = series["XAU"]["ret"] if "XAU" in series else None
wti_ret = series["WTI"]["ret"] if "WTI" in series else None

# ---------- candidate factor computation (per asset own calendar) ----------
cands = {}
for s, d in series.items():
    close, ret = d["close"], d["ret"]
    vol20, vol60 = d["vol20"], d["vol60"]
    cands.setdefault("zsco_20", {})[s] = (close / d["sma20"] - 1.0) / vol20
    cands.setdefault("zsco_40", {})[s] = (close / d["sma40"] - 1.0) / vol60
    cands.setdefault("autocorr_20", {})[s] = ret.rolling(20).apply(
        lambda x: pd.Series(x).autocorr(1) if np.isfinite(x).sum() >= 10 else np.nan, raw=False)
    cands.setdefault("drawdown_60", {})[s] = close / d["max60"] - 1.0
    cands.setdefault("hurst_60", {})[s] = hurst_rs(d["logp"], 60)
    cands.setdefault("relstrength_40_med", {})[s] = close / close.shift(40) - 1.0
    # global risk-on beta (asset ret regressed on EW index ret)
    r_al = ret.reindex(GRID)
    beta = rolling_beta(ew_ret_ser, r_al, 60)
    cands.setdefault("ewidx_beta_60", {})[s] = beta
    # correlation to EW index (systematic-ness)
    corr = rolling_corr(ew_ret_ser, r_al, 60)
    cands.setdefault("ewidx_corr_60", {})[s] = corr
    # XAU beta (safe haven)
    if xau_ret is not None:
        xa = xau_ret.reindex(GRID)
        cands.setdefault("beta_xau_60", {})[s] = rolling_beta(xa, r_al, 60)
    # WTI beta (energy)
    if wti_ret is not None:
        wt = wti_ret.reindex(GRID)
        cands.setdefault("beta_wti_60", {})[s] = rolling_beta(wt, r_al, 60)
    # Amihud illiquidity
    vol = d["volume"] if "volume" in d else None
    if vol is not None:
        illiq = (ret.abs() / (vol + 1e-9)).rolling(20).mean()
        cands.setdefault("amihud_20", {})[s] = illiq
        v5 = vol.rolling(5).mean(); v60 = vol.rolling(60).mean()
        cands.setdefault("volume_z_5x60", {})[s] = v5 / v60
    # vol z-score (cross-sectional, computed below on grid)
    cands.setdefault("vol20_raw", {})[s] = vol20
    cands.setdefault("skew60", {})[s] = rolling_skew(ret, 60)

# yield-spread beta (needs aligned spread; build on grid)
if yspread is not None:
    ysp_g = yspread.reindex(GRID)
    dsp = ysp_g.diff()
    for s in series:
        r_al = series[s]["ret"].reindex(GRID)
        cands.setdefault("beta_yspread_60", {})[s] = rolling_beta(dsp, r_al, 60)

# cross-sectional transforms on grid
def xsec_zscore(mat):
    out = np.full_like(mat, np.nan)
    for t in range(mat.shape[0]):
        row = mat[t]
        ok = ~np.isnan(row)
        if ok.sum() >= 8:
            m = np.nanmean(row[ok]); sd = np.nanstd(row[ok])
            if sd > 1e-12:
                out[t, ok] = (row[ok] - m) / sd
    return out

vol20_mat = to_grid(cands.pop("vol20_raw"))
cands["vol_zscore_20"] = {s: pd.Series(xsec_zscore(vol20_mat)[:, i], index=GRID)
                          for i, s in enumerate(ASSETS)}

# relative strength vs cross-sectional median (grid)
rs40 = to_grid(cands.pop("relstrength_40_med"))
rs40c = np.full_like(rs40, np.nan)
for t in range(rs40.shape[0]):
    row = rs40[t]
    ok = ~np.isnan(row)
    if ok.sum() >= 8:
        med = np.nanmedian(row[ok])
        rs40c[t, ok] = row[ok] - med
cands["relstrength_40_med"] = {s: pd.Series(rs40c[:, i], index=GRID)
                               for i, s in enumerate(ASSETS)}

print(f"candidates: {sorted(cands.keys())}")

# ---------- evaluate ----------
fwd = fwd_by_horizon_dict(series)
dates = np.array(GRID)
results = {}

def report(name, mat):
    rank_mat = cross_sectional_rank(mat)
    ics = spearman_ic_matrix(mat, fwd[10])
    summ = summarize(ics, dates, name, HORIZON)
    if summ is None:
        print(f"{name}: NO VALID IC DATES")
        return None
    cov_ad, cov_d8 = coverage_stats(mat)
    to_ = turnover_10d_rank(rank_mat)
    dec = decay_curve(mat, fwd)
    corrs, mx_name, mx_abs = library_pairwise_corr(mat)
    ic, icir = summ["ic"], summ["icir"]
    ok = (abs(ic) >= 0.007) and (abs(icir) >= 0.084)
    corr_ok = mx_abs < 0.5
    print("=" * 100)
    print(f"{name}: IC={ic:+.4f} ICIR={icir:+.4f} hit={summ['hit']:.3f} n={summ['n_ic_dates']} "
          f"cov_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} turn={to_:.3f} maxlibcorr={mx_abs:.3f} ({mx_name})")
    print(f"   regime: { {k: v['ic'] for k, v in summ['regime'].items()} }")
    print(f"   decay10: {dec}  GATE {'PASS' if ok else 'FAIL'} (ic{'+' if ok else '-'} icir{'+' if (abs(icir)>=0.084) else '-'}) corr_ok={corr_ok}")
    return {"ic": ic, "icir": icir, "hit": summ["hit"], "n": summ["n_ic_dates"],
            "cov_ad": cov_ad, "cov_d8": cov_d8, "turn": to_, "maxlibcorr": mx_abs,
            "maxlibname": mx_name, "regime": summ["regime"], "decay": dec,
            "gate_ok": ok and corr_ok}

for name, cd in cands.items():
    mat = to_grid(cd)
    results[name] = report(name, mat)

print("\n===== SUMMARY =====")
for k, v in results.items():
    if v is None:
        continue
    print(f"{k:24s} IC={v['ic']:+.4f} ICIR={v['icir']:+.4f} hit={v['hit']:.3f} n={v['n']} "
          f"turn={v['turn']:.3f} maxlib={v['maxlibcorr']:.3f} GATE={'PASS' if v['gate_ok'] else '--'}")

with open("scripts/miner_1_20260910_batchA_results.json", "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "regime"} for k, v in results.items() if v}, f, indent=1, default=str)
print("results saved to scripts/miner_1_20260910_batchA_results.json")
