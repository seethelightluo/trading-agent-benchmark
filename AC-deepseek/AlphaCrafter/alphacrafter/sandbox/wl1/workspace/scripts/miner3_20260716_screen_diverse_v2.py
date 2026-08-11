"""Miner3 screen: diverse factor families vs library correlation.
Goal: find factors with |IC1|>=0.007, |ICIR1|>=0.084 AND max pooled Spearman
rho vs recoverable library artifacts < 0.5 (gate contract).
Validation window 2021-01-01..2026-07-15, daily rank IC on 15-name panel.
"""
import os, json, pickle, base64, gzip, zlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SYMBOLS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
           "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
cache = pickle.load(open("scripts/panel_cache.pkl", "rb"))
close, open_, high, low, vol = cache["close"], cache["open"], cache["high"], cache["low"], cache["vol"]
ret = cache["ret"]
macro = cache["macro"]  # index aligned with panel? check
idx = close.index

# macro: align to panel index (may be shorter/offset)
print("macro cols:", list(macro.columns), macro.shape)
macro_reindexed = macro.reindex(idx).ffill()

VALID_START = pd.Timestamp("2021-01-01")
VALID_END = pd.Timestamp("2026-07-15")
mask_dates = (idx >= VALID_START) & (idx <= VALID_END)

# ---------------------------------------------------------------- library artifacts
def load_npy(name):
    p = os.path.join("factors", name)
    if not os.path.exists(p):
        return None
    return pd.DataFrame(np.load(p), index=idx, columns=SYMBOLS)

lib_artifacts = {}
for npy in ["miner2_20260716_mom_10d_skip5.npy", "miner1_20260716_er20.npy",
            "miner1_20260716_rev5x_er_soft.npy", "miner2_20260716_nclv_1d.npy"]:
    df = load_npy(npy)
    if df is not None and len(df) > 100:
        lib_artifacts[npy.replace(".npy", "")] = df
print("library artifacts loaded:", list(lib_artifacts.keys()))

def pooled_spearman(a, b):
    A = a.values.astype(float).ravel(); B = b.values.astype(float).ravel()
    m = np.isfinite(A) & np.isfinite(B)
    if m.sum() < 30:
        return np.nan
    return float(spearmanr(A[m], B[m]).statistic)

def max_lib_rho(sig):
    rhos = {k: abs(pooled_spearman(sig, v)) for k, v in lib_artifacts.items()}
    rhos = {k: v for k, v in rhos.items() if np.isfinite(v)}
    if not rhos:
        return np.nan, {}
    k = max(rhos, key=rhos.get)
    return rhos[k], {kk: round(vv, 3) for kk, vv in rhos.items()}

# ---------------------------------------------------------------- factor library helpers
def zscore_ts(s):
    mu = s.rolling(120, min_periods=30).mean()
    sd = s.rolling(120, min_periods=30).std()
    return (s - mu) / sd.replace(0, np.nan)

def roll_beta(y, x, win):
    """rolling OLS slope of y on x, per column, window win"""
    out = pd.DataFrame(np.nan, index=y.index, columns=y.columns)
    yv, xv = y.values, x.values
    for i in range(win, len(yv)):
        ys, xs = yv[i - win:i], xv[i - win:i]
        xm = np.nanmean(xs, axis=0); ym = np.nanmean(ys, axis=0)
        num = np.nansum((xs - xm) * (ys - ym), axis=0)
        den = np.nansum((xs - xm) ** 2, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            b = np.where(den > 1e-12, num / np.where(den > 1e-12, den, np.nan), np.nan)
        out.iloc[i] = b
    return out

def roll_autocorr(x, lag, win):
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    yv = x.values
    for i in range(win + lag, len(yv)):
        seg = yv[i - win:i]
        a, b = seg[:-lag], seg[lag:]
        m = np.isfinite(a) & np.isfinite(b)
        out.iloc[i] = np.where(m.sum(axis=0) > win * 0.5,
                               [np.corrcoef(a[mm], b[mm])[0, 1] if mm.sum() > 5 else np.nan
                                for mm in m.T], np.nan)
    return out

# ---------------------------------------------------------------- candidate factors
F = {}

# F1: beta to dVIX (60d) - negative direction expected (risk-off)
F["beta_dVIX_60"] = roll_beta(ret, macro_reindexed["VIX"].pct_change(), 60)

# F2: beta to dDXY (60d)
F["beta_dDXY_60"] = roll_beta(ret, macro_reindexed["DXY"].pct_change(), 60)

# F3: beta to dUSDJPY (60d)
F["beta_dJPY_60"] = roll_beta(ret, macro_reindexed["USDJPY"].pct_change(), 60)

# F4: vol term structure: rv5 / rv60
rv5 = ret.rolling(5).std(); rv60 = ret.rolling(60).std()
F["vol_term_5_60"] = rv5 / rv60.replace(0, np.nan)

# F5: skewness 60d
F["skew_60"] = ret.rolling(60).skew()

# F6: downside/upside semi-vol ratio (30d)
def semivol_ratio(r, win=30):
    out = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
    v = r.values
    for i in range(win, len(v)):
        seg = v[i - win:i]
        down = np.sqrt(np.nanmean(np.minimum(seg, 0) ** 2, axis=0))
        up = np.sqrt(np.nanmean(np.maximum(seg, 0) ** 2, axis=0))
        out.iloc[i] = down / np.where(up > 1e-12, up, np.nan)
    return out
F["semivol_down_up_30"] = semivol_ratio(ret, 30)

# F7: autocorrelation 5d lag, 40d window
F["autocorr5_40"] = roll_autocorr(ret, 5, 40)

# F8: market beta 60d (beta vs equal-weight cross-sectional return)
ew = ret.mean(axis=1)
F["beta_mkt_60"] = roll_beta(ret, ew, 60)

# F9: Amihud illiquidity z: |ret|/volume, 20d mean, z-scored (volume may be sparse)
with np.errstate(divide="ignore", invalid="ignore"):
    amihud = (ret.abs() / vol.replace(0, np.nan))
F["amihud_z_20"] = zscore_ts(amihud.rolling(20, min_periods=10).mean())

# F10: Monday effect: mean return on Mondays over last 20 weeks vs overall
def weekday_mean_ratio(r, target=0, win_weeks=20):
    out = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
    d = r.index.dayofweek.values
    v = r.values
    wd = (d == target)
    # rolling sum over last win_weeks*5 days
    win = win_weeks * 5
    for i in range(win, len(v)):
        seg = v[i - win:i]
        wmask = wd[i - win:i]
        if wmask.sum() < 4:
            continue
        mon = np.nanmean(seg[wmask], axis=0)
        allm = np.nanmean(seg, axis=0)
        out.iloc[i] = mon - allm
    return out
F["monday_eff_20w"] = weekday_mean_ratio(ret, 0, 20)

# F11: Friday effect
F["friday_eff_20w"] = weekday_mean_ratio(ret, 4, 20)

# F12: crypto beta 60d (beta vs BTC ret)
F["beta_btc_60"] = roll_beta(ret, ret["BTC"], 60)

# F13: range position 5d avg ((close-low)/(high-low)), likely conflicts; keep as control
def range_pos(rng_win=5):
    out = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for c in SYMBOLS:
        df = pd.concat([open_[c], close[c], high[c], low[c]], axis=1).dropna()
        rng = (df.iloc[:, 2] - df.iloc[:, 3]).replace(0, np.nan)
        pos = ((df.iloc[:, 1] - df.iloc[:, 3]) / rng).rolling(rng_win).mean()
        out.loc[df.index, c] = pos
    return out
F["range_pos_5d"] = range_pos(5)

# F14: Parkinson vol 20d z
park = np.sqrt(np.log(2) ** -1 * (np.log(high / low) ** 2).rolling(20).mean())
F["parkvol_z_20"] = zscore_ts(park)

# F15: volume z-score 20d
F["volz_20"] = zscore_ts(vol.rolling(20, min_periods=10).mean())

# F16: upper shadow ratio 20d: mean((high-max(o,c))/close)
def shadow_ratio():
    out = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for c in SYMBOLS:
        df = pd.concat([open_[c], close[c], high[c], low[c]], axis=1).dropna()
        o, cl, h, l = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 3]
        up = ((h - np.maximum(o, cl)) / cl.replace(0, np.nan)).rolling(20).mean()
        dn = ((np.minimum(o, cl) - l) / cl.replace(0, np.nan)).rolling(20).mean()
        out.loc[df.index, c] = up - dn
    return out
F["shadow_bal_20"] = shadow_ratio()

# F17: momentum acceleration: 20d ret minus 10d ret (change in short trend)
F["mom_acc_20_10"] = ret.rolling(20).sum() - ret.rolling(10).sum()

# F18: 5d realized vol z-score
F["rv5_z"] = zscore_ts(ret.rolling(5).std())

# F19: trend consistency 20d: |sum ret| / sum|ret|
def trend_consist(win=20):
    s = ret.rolling(win).sum()
    a = ret.abs().rolling(win).sum().replace(0, np.nan)
    return s / a
F["trend_consist_20"] = trend_consist(20)

# F20: overnight ratio z (gap / range)
def overnight_ratio(win=20):
    out = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    prev_close = close.shift(1)
    gap = open_ - prev_close
    rng = (high - low).replace(0, np.nan)
    return (gap / rng).rolling(win).mean()
F["overnight_ratio_20"] = overnight_ratio(20)

# F21: kurtosis 60d
F["kurt_60"] = ret.rolling(60).kurt()

# F22: 120d return (slow momentum, control - likely conflicts)
F["mom_120d"] = ret.rolling(120).sum()

# ---------------------------------------------------------------- validation
fwd = ret.shift(-1)  # next-day return
print("\n%-22s %8s %8s %6s %6s %7s %7s | %s" %
      ("factor", "ic1", "icir1", "hit", "cov", "n_dates", "turn10", "maxrho(lib)"))
results = {}
for name, sig in F.items():
    s = sig.reindex(idx)
    sub_s = s[mask_dates]
    sub_f = fwd[mask_dates]
    ics, hits, n_obs = [], [], 0
    for i in range(len(sub_s)):
        x = sub_s.iloc[i].values.astype(float)
        y = sub_f.iloc[i].values.astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 8:
            continue
        r = spearmanr(x[m], y[m]).statistic
        if np.isfinite(r):
            ics.append(r)
            hits.append(1 if r > 0 else 0)
            n_obs += int(m.sum())
    if len(ics) < 60:
        print("%-22s %8s insufficient dates (%d)" % (name, "", len(ics)))
        results[name] = None
        continue
    ic1 = float(np.mean(ics)); icir1 = float(np.mean(ics) / np.std(ics) * np.sqrt(len(ics)))
    hit = float(np.mean(hits)); n_dates = len(ics)
    cov = n_obs / (n_dates * len(SYMBOLS))
    # turnover: mean abs change of cross-sectional rank over 10d (normalized)
    ranks = s.rank(axis=1)
    r10 = ranks.diff(10).abs().mean(axis=1)
    turn10 = float(r10[mask_dates].mean() / (len(SYMBOLS) - 1))
    mr, rhos = max_lib_rho(s)
    results[name] = dict(ic1=ic1, icir1=icir1, hit=hit, cov=cov, n_dates=n_dates,
                         turn10=turn10, maxrho=mr, rhos=rhos)
    gate = "PASS" if (abs(ic1) >= 0.007 and abs(icir1) >= 0.084 and mr < 0.5) else ""
    print("%-22s %8.4f %8.4f %6.3f %6.2f %7d %7.3f | %5.3f %s" %
          (name, ic1, icir1, hit, cov, n_dates, turn10, mr, gate))

json.dump({k: (v if v is None else {kk: (round(vv, 4) if isinstance(vv, float) else vv)
            for kk, vv in v.items()}) for k, v in results.items()},
          open("scripts/miner3_diverse_v2_results.json", "w"), indent=1)
print("\nsaved scripts/miner3_diverse_v2_results.json")
