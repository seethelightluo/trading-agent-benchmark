"""miner_2 (2032-05-31) - re-validate effective factors + explore new candidates.

Visible data through 2032-05-28 (previous completed trading day).
Universe: 15 tradable cross-asset instruments (cross-section of 15, not stocks).
Gates: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10 (benchmark-wide admission).
Reports decay, coverage, turnover, max_abs_library_correlation, recent drift,
yearly splits. No live-account interaction (miner only).
Fixes: pandas fillna(method=...) deprecation -> .ffill(); robust ADX true-range.
"""
import sys, time, warnings, json, zlib, base64, hashlib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, "scripts")
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, TRADABLE)

t0 = time.time()
panels = load_panels(days=4000)
closes = close_panel(panels)
rets = closes.pct_change()
mkt_ret = rets.mean(axis=1)
print(f"closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()} | {time.time()-t0:.1f}s", flush=True)


def align(series, idx):
    return series.reindex(idx).ffill()


vix = align(panels["VIX"]["close"].astype(float), closes.index)
dxy = align(panels["DXY"]["close"].astype(float), closes.index)
usdjpy = align(panels["USDJPY"]["close"].astype(float), closes.index)
usdcny = align(panels["USDCNY"]["close"].astype(float), closes.index)
eurusd = align(panels["EURUSD"]["close"].astype(float), closes.index)

hi = pd.concat({a: panels[a]["high"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)
lo = pd.concat({a: panels[a]["low"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)
vol_panel = pd.concat({a: panels[a]["volume"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)
op = pd.concat({a: panels[a]["open"].astype(float) for a in TRADABLE if a in panels}, axis=1).sort_index().reindex(closes.index)

H = 10
fwd = forward_returns(closes, H)
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()


def rolling_beta(y, x, win=60, min_obs=40):
    out = {}
    for a in y.columns:
        z = pd.concat([y[a].rename("y"), x.rename("x")], axis=1).dropna()
        cov = z["y"].rolling(win).cov(z["x"])
        var = z["x"].rolling(win).var()
        b = (cov / var).where(z["x"].rolling(win).count() >= min_obs)
        out[a] = b
    return pd.DataFrame(out, index=y.index)


def rolling_corr_series(a, b, win=20, min_obs=15):
    out = {}
    for col in a.columns:
        z = pd.concat([a[col].rename("a"), b.rename("b")], axis=1).dropna()
        c = z["a"].rolling(win).corr(z["b"])
        out[col] = c.where(z["b"].rolling(win).count() >= min_obs)
    return pd.DataFrame(out, index=a.index)


# ---------- library reference signals for correlation audit ----------
lib = {}
lib["vol_adj_mom_accel_20x60"] = (closes/closes.shift(20)-1 - (closes/closes.shift(60)-1)) / rets.rolling(20).std()
lib["dn_mkt_beta_60d"] = rolling_beta(rets, mkt_ret.clip(upper=0), 60)
lib["rate_beta_cn10y_60d"] = rolling_beta(rets, closes["CN10Y"].pct_change(), 60)
lib["mom_10d_skip5"] = closes.shift(5) / closes.shift(15) - 1.0
lib["mom_120d_skip5"] = closes.shift(5) / closes.shift(125) - 1.0
lib["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
lib["vol_ratio_20_60"] = vol20 / vol60
lib["volume_z_20"] = (vol_panel - vol_panel.rolling(60).mean()) / (vol_panel.rolling(60).std() + 1e-12)
lib["usdcny_beta_60d"] = rolling_beta(rets, usdcny.pct_change(), 60)
lib["eurusd_beta_60d"] = rolling_beta(rets, eurusd.pct_change(), 60)
lib["vix_beta_cond_60x20"] = rolling_beta(rets, vix.pct_change(), 60) * (vix.pct_change().rolling(20).mean() > 0).astype(float)
lib["mom_vs_median_60d"] = (closes/closes.shift(60)-1) - (closes/closes.shift(60)-1).rolling(60).median()
lib["us10y_cond_beta_60d"] = rolling_beta(rets, closes["US10Y"].pct_change(), 60) * (closes["US10Y"].pct_change().rolling(60).mean() > 0).astype(float)
lib["downside_ratio_60d"] = rets.clip(upper=0).rolling(60).std() / (rets.rolling(60).std() + 1e-12)
lib["max_dd_60d"] = closes / closes.rolling(60).max() - 1.0
lib["kurt_60d"] = rets.rolling(60).kurt()
lib["corr_asset_mkt_20"] = rolling_corr_series(rets, mkt_ret, 20)
lib["hl_pos_20d"] = (hi - lo) / (closes + 1e-12)
lib["xau_beta_60"] = rolling_beta(rets, rets["XAU"], 60)
lib["comm_beta_60"] = rolling_beta(rets, rets[["WTI", "XAU", "COPPER"]].mean(axis=1), 60)
lib["spread_beta_cnus_60"] = rolling_beta(rets, (closes["CN10Y"] - closes["US10Y"]).diff(), 60)
lib["updown_vol_ratio_20"] = rets.clip(lower=0).rolling(20).std() / (rets.clip(upper=0).rolling(20).std() + 1e-12)
lib["trend_tstat_60"] = (closes/closes.shift(60)-1) / (vol60 * np.sqrt(60) + 1e-12)
lib["crypto_beta_60"] = rolling_beta(rets, rets[["BTC", "ETH"]].mean(axis=1), 60)
lib["max_gain_20"] = rets.rolling(20).max()
lib["skew_60d"] = rets.rolling(60).skew()
lib["beta_btc_60d"] = rolling_beta(rets, closes["BTC"].pct_change(), 60)
lib["vix_level_z"] = pd.DataFrame({a: ((vix - vix.rolling(60).mean()) / (vix.rolling(60).std() + 1e-12)).values for a in closes.columns}, index=closes.index)
lib["dxy_mom20"] = pd.DataFrame({a: (dxy/dxy.shift(20) - 1).values for a in closes.columns}, index=closes.index)
lib["path_efficiency_20"] = (closes/closes.shift(20)-1).abs() / (rets.abs().rolling(20).sum() + 1e-12)
lib["win_rate_20"] = (rets > 0).rolling(20).mean()
lib["range_pos_20"] = (closes - lo.rolling(20).min()) / (hi.rolling(20).max() - lo.rolling(20).min() + 1e-12)
lib["close_z_20"] = (closes - closes.rolling(20).mean()) / (closes.rolling(20).std() + 1e-12)
lib["xau_rel_mom_20"] = (closes/closes.shift(20)-1).sub((closes["XAU"]/closes["XAU"].shift(20)-1), axis=0)
lib["us10y_corr_20"] = rolling_corr_series(rets, closes["US10Y"].pct_change(), 20)
lib["gap_intensity_20"] = ((op / closes.shift(1) - 1).abs()).rolling(20).mean()
lib["vol_spike_5_60"] = rets.rolling(5).std() / (vol60 + 1e-12)
print(f"library refs: {len(lib)} | {time.time()-t0:.1f}s", flush=True)

# ---------- fast spearman-based library correlation ----------
ranked_all = {}


def get_ranked(name, sig):
    if name not in ranked_all:
        ranked_all[name] = sig.rank(axis=0)
    return ranked_all[name]


def pairwise_max_abs_corr(cand, libsig, cand_name, lib_name):
    a = get_ranked(cand_name, cand)
    b = get_ranked(lib_name, libsig)
    cols = a.columns.intersection(b.columns)
    best = 0.0
    for c in cols:
        x = a[c].dropna(); y = b[c].dropna()
        idx = x.index.intersection(y.index)
        if len(idx) < 120:
            continue
        xv = x.loc[idx].to_numpy(dtype=float); yv = y.loc[idx].to_numpy(dtype=float)
        if xv.std() < 1e-12 or yv.std() < 1e-12:
            continue
        r = np.corrcoef(xv, yv)[0, 1]
        if not np.isnan(r) and abs(r) > best:
            best = abs(r)
    return best


def full_report(name, sig, expected_sign=1):
    ics = rank_ic_series(sig, fwd)
    s = summarize_ic(ics, expected_sign=expected_sign)
    cov = coverage_metrics(sig)
    to = turnover_rank(sig, 10)
    dec = decay_profile(sig, closes, horizons=[1, 2, 3, 5, 10, 20])
    mc = {k: pairwise_max_abs_corr(sig, v, name, k) for k, v in lib.items()}
    mc = {k: v for k, v in mc.items() if v == v}
    mcl = max(mc.values()) if mc else np.nan
    mcl_name = max(mc, key=mc.get) if mc else ""
    r63 = ics.iloc[-63:].mean() if len(ics) >= 63 else np.nan
    r126 = ics.iloc[-126:].mean() if len(ics) >= 126 else np.nan
    r252 = ics.iloc[-252:].mean() if len(ics) >= 252 else np.nan
    yr = {}
    for y in range(2020, 2033):
        sub = ics[(ics.index.year == y)]
        if len(sub) > 20:
            mm, ss = sub.mean(), sub.std(ddof=1)
            yr[y] = (round(float(mm), 4), round(float(mm/ss), 2) if ss and ss > 0 else np.nan)
    flag = "  <== FULL-PASS" if (abs(s["ic"]) >= 0.0070 and abs(s["icir"]) >= 0.0840) else ""
    print(f"\n=== {name} (es={expected_sign:+d}){flag} ===", flush=True)
    print(f"  IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['ic_hit_ratio']:.2f} n={s['n_ic_dates']} "
          f"std={s['ic_std']:.3f} | cov8={cov['coverage_dates_ge8']:.2f} cov_asset={cov['coverage_asset_days']:.2f} "
          f"to10={to if to is not None else float('nan'):.2f}", flush=True)
    print(f"  decay(1,2,3,5,10,20)={[round(x,4) if x==x else None for x in dec.values()]} | "
          f"r63={r63:+.4f} r126={r126:+.4f} r252={r252:+.4f}", flush=True)
    print(f"  max_abs_lib_corr={mcl:.3f} (vs {mcl_name})", flush=True)
    print(f"  yearly_ic(mean,icir): {yr}", flush=True)
    return s, ics, sig, mc


# ============ 1) RE-VALIDATE current effective factors ============
print("\n\n########## RE-VALIDATION (current effective) ##########", flush=True)
existing = {
    "vol_adj_mom_accel_20x60": ((closes/closes.shift(20)-1 - (closes/closes.shift(60)-1)) / rets.rolling(20).std(), 1),
    "dn_mkt_beta_60d": (rolling_beta(rets, mkt_ret.clip(upper=0), 60), 1),
    "rate_beta_cn10y_60d": (rolling_beta(rets, closes["CN10Y"].pct_change(), 60), -1),
}
rev = {}
for name, (sig, es) in existing.items():
    s, ics, _, _ = full_report(name, sig, expected_sign=es)
    rev[name] = s

# ============ 2) NEW CANDIDATES ============
print("\n\n########## NEW CANDIDATES ##########", flush=True)
C = {}
# A: Kaufman efficiency ratio 60d (trend purity, long horizon)
C["kaufman_eff_60"] = (closes - closes.shift(60)).abs() / (rets.abs().rolling(60).sum() + 1e-12)
# B: ADX-style trend strength 14d (directional movement index)
up = hi.diff(); dn = -lo.diff()
plus_dm = pd.DataFrame(np.where((up > dn) & (up > 0), up, 0.0), index=closes.index, columns=closes.columns)
minus_dm = pd.DataFrame(np.where((dn > up) & (dn > 0), dn, 0.0), index=closes.index, columns=closes.columns)
tr_hl = hi - lo
tr_hc = (hi - closes.shift(1)).abs()
tr_lc = (lo - closes.shift(1)).abs()
tr = pd.concat([tr_hl, tr_hc, tr_lc]).groupby(level=0).max()
tr = tr.reindex(closes.index).ffill().fillna(0.0)
atr = tr.rolling(14).mean() + 1e-12
pdi = 100.0 * plus_dm.rolling(14).mean() / atr
mdi = 100.0 * minus_dm.rolling(14).mean() / atr
dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi + 1e-12)
C["adx_14"] = dx.rolling(14).mean()
# C: 20d Sharpe-like t-stat (risk-adjusted momentum)
C["tstat_20"] = rets.rolling(20).mean() / (vol20 + 1e-12)
# D: inverse volatility 20d (low-vol premium)
C["inv_vol_20"] = 1.0 / (vol20 + 1e-12)
# E: momentum quality: sum(up)/|sum(down)| 20d
up_ret = rets.clip(lower=0); dn_ret = rets.clip(upper=0)
C["mom_quality_20"] = up_ret.rolling(20).sum() / (dn_ret.rolling(20).sum().abs() + 1e-12)
# F: range-based volatility 20d (intraday range / close)
C["range_vol_20"] = ((hi - lo) / (closes + 1e-12)).rolling(20).mean()
# G: idiosyncratic vol 20d (residual std vs mkt)
beta20 = rolling_beta(rets, mkt_ret, 20)
resid = rets - beta20 * mkt_ret
C["resid_vol_20"] = resid.rolling(20).std()
# H: plain 20d market beta
C["beta_mkt_20"] = beta20
# I: 20d skewness
C["skew_20"] = rets.rolling(20).skew()
# J: distance from 200d SMA
C["dist_sma200"] = closes / closes.rolling(200).mean() - 1.0
# K: overnight/gap direction persistence: mean open gap over 20d
C["overnight_mom_20"] = (op / closes.shift(1) - 1.0).rolling(20).mean()
# L: gold-relative 60d momentum (safe-haven rotation, long horizon)
C["xau_rel_mom_60"] = (closes/closes.shift(60)-1).sub((closes["XAU"]/closes["XAU"].shift(60)-1), axis=0)
# M: WTI-relative 20d momentum (energy-relative rotation)
C["wti_rel_mom_20"] = (closes/closes.shift(20)-1).sub((closes["WTI"]/closes["WTI"].shift(20)-1), axis=0)
# N: BTC-relative 20d momentum (crypto-relative rotation)
C["btc_rel_mom_20"] = (closes/closes.shift(20)-1).sub((closes["BTC"]/closes["BTC"].shift(20)-1), axis=0)
# O: 10d/60d vol ratio (smooth vol acceleration)
C["vol_ratio_10_60"] = rets.rolling(10).std() / (vol60 + 1e-12)
# P: 60d return dispersion of asset vs cross-sectional median (relative strength)
C["rel_strength_60"] = (closes/closes.shift(60)-1) - (closes/closes.shift(60)-1).median(axis=1)
# Q: time-to-recover / drawdown depth ratio (mean-reversion potential) 20d
C["dd_recover_20"] = (closes / closes.rolling(20).max() - 1.0)
# R: BTC/ETH vs SPX relative momentum 20d (crypto risk appetite)
C["crypto_risk_app_20"] = (closes[["BTC", "ETH"]].mean(axis=1) / closes[["BTC", "ETH"]].mean(axis=1).shift(20) - 1.0) \
    - (closes["SPX"] / closes["SPX"].shift(20) - 1.0)
C["crypto_risk_app_20"] = pd.DataFrame({a: C["crypto_risk_app_20"].values for a in closes.columns}, index=closes.index)
# S: 5d momentum reversal (short-term contrarian)
C["mom_rev_5"] = closes.shift(1) / closes.shift(6) - 1.0

signs = {k: -1 for k in ["inv_vol_20", "range_vol_20", "resid_vol_20", "dd_recover_20", "mom_rev_5"]}
results = {}
for name, sig in C.items():
    es = signs.get(name, 1)
    s, ics, sig, mc = full_report(name, sig, expected_sign=es)
    results[name] = (s, ics, sig, mc)

passing = {k: v for k, v in results.items() if abs(v[0]["ic"]) >= 0.0070 and abs(v[0]["icir"]) >= 0.0840}
print(f"\n\nFULL-PASS count (new candidates): {len(passing)}", flush=True)
for k in passing:
    mc = passing[k][3]
    mcl = max(mc.values()) if mc else np.nan
    print(f"  PASS: {k} IC={passing[k][0]['ic']:+.4f} ICIR={passing[k][0]['icir']:+.3f} "
          f"max_abs_lib_corr={mcl:.3f}", flush=True)

print(f"\ntotal time {time.time()-t0:.1f}s", flush=True)
