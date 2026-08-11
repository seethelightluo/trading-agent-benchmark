"""miner_3 batch B screen: novel factor candidates (skew/tail, drawdown, vol-ratio,
range position, serial corr, upside beta, vol-adj momentum, volume trend, crypto/gold
beta, mkt corr, relative momentum, max loss, hilo range, short reversal).

Admission gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10 (15-asset universe).
Library correlation check vs all 7 persisted factors.
"""
import sys, time, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 coverage_metrics, turnover_rank, library_signals,
                                 max_library_corr)

panels = load_panels(days=3000)
closes = close_panel(panels)
rets = closes.pct_change()
H_ADM = 10
HORIZONS = (1, 3, 5, 10, 20)

# ---- recompute ALL 7 library factor signals for correlation check ----
def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40, exclude_self=None):
    beta = {}
    for a in asset_ret.columns:
        if exclude_self is not None and a == exclude_self:
            beta[a] = pd.Series(np.nan, index=asset_ret.index)
            continue
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        cov = z["a"].rolling(win).cov(z["m"])
        var = z["m"].rolling(win).var()
        b = (cov / var).where(z["m"].rolling(win).count() >= min_obs)
        beta[a] = b
    return pd.DataFrame(beta, index=asset_ret.index)

lib = {}
mkt = rets.mean(axis=1)
dn = mkt.where(mkt < 0).fillna(0.0)
lib["mom_10d_skip5"] = closes.shift(5) / closes.shift(15) - 1.0
lib["mom_120d_skip5"] = closes.shift(5) / closes.shift(125) - 1.0
lib["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
lib["dn_mkt_beta_60d"] = rolling_beta(rets, dn, 60, 40)
vix = panels["VIX"]["close"].astype(float)
vix_ret = vix.pct_change()
vix_beta = rolling_beta(rets, vix_ret, 60, 40)
lib["vix_beta_cond_60x20"] = -vix_beta * (vix / vix.shift(20) - 1.0)
lib["eurusd_beta_60d"] = rolling_beta(rets, panels["EURUSD"]["close"].astype(float).pct_change(), 60, 40)
lib["rate_beta_cn10y_60d"] = rolling_beta(rets, panels["CN10Y"]["close"].astype(float).pct_change(), 60, 40)
for k in lib:
    lib[k] = lib[k].reindex(closes.index)

# ---- vectorized rank IC ----
def rank_ic_series_vec(factor_panel, fwd, min_valid=8):
    fr = factor_panel.rank(axis=1, method="average")
    rr = fwd.rank(axis=1, method="average")
    count = (fr.notna() & rr.notna()).astype(float)
    n = count.sum(axis=1)
    fm = fr.fillna(0.0) - (fr.fillna(0.0) * count).sum(axis=1) / n.replace(0, np.nan)
    rm = rr.fillna(0.0) - (rr.fillna(0.0) * count).sum(axis=1) / n.replace(0, np.nan)
    fm = fm.where(count > 0)
    rm = rm.where(count > 0)
    num = (fm * rm).sum(axis=1)
    den = np.sqrt((fm ** 2).sum(axis=1) * (rm ** 2).sum(axis=1))
    ic = (num / den.replace(0, np.nan)).where((n >= min_valid) & (den > 1e-14))
    return ic.dropna().rename("ic")

def summarize_ic_vec(ics, expected_sign=1):
    ic = float(ics.mean())
    std = float(ics.std(ddof=1))
    icir = ic / std if std > 0 else 0.0
    hit = float((np.sign(ics) == expected_sign).mean())
    return {"ic": round(ic, 4), "icir": round(icir, 4), "ic_hit_ratio": round(hit, 3),
            "n_ic_dates": int(len(ics)), "ic_std": round(std, 4)}

def per_asset(func):
    out = {}
    for a in closes.columns:
        s = closes[a].dropna()
        out[a] = func(s)
    return pd.DataFrame(out).reindex(closes.index)

def rolling_skew(s, win=60):
    return s.rolling(win).skew()

def rolling_kurt(s, win=60):
    return s.rolling(win).kurt()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0.0).rolling(n).mean()
    dn = (-d.clip(upper=0.0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)

cands = {}
# tail / asymmetry
cands["skew_60d"] = per_asset(lambda s: rolling_skew(s.pct_change(), 60))
cands["kurt_60d"] = per_asset(lambda s: rolling_kurt(s.pct_change(), 60))
# drawdown
cands["drawdown_60d"] = per_asset(lambda s: s / s.rolling(60).max() - 1.0)
# vol regime ratio
cands["vol_ratio_10x60"] = per_asset(lambda s: s.pct_change().rolling(10).std() / s.pct_change().rolling(60).std())
# intraday range position persistence
cands["range_pos_20d"] = per_asset(lambda s: ((s - s.rolling(2).min()) / (s.rolling(2).max() - s.rolling(2).min())).rolling(20).mean())
# serial correlation (1-day lag, 10d window)
cands["serial_corr_10d"] = per_asset(lambda s: s.pct_change().rolling(10).apply(lambda x: x.autocorr() if len(x) >= 8 else np.nan, raw=False))
# upside beta (complement of dn beta)
up = mkt.where(mkt > 0).fillna(0.0)
cands["up_side_beta_60d"] = rolling_beta(rets, up, 60, 40)
# vol-adjusted momentum (sharpe-like 60d)
cands["vol_adj_mom_60d"] = per_asset(lambda s: s.pct_change().rolling(60).mean() / s.pct_change().rolling(60).std())
# volume trend
vol_panel = pd.concat({a: panels[a]["volume"].astype(float) for a in closes.columns}, axis=1).reindex(closes.index)
cands["volume_trend_20x60"] = vol_panel.rolling(20).mean() / vol_panel.rolling(60).mean()
# crypto beta
cands["crypto_beta_60d"] = rolling_beta(rets, rets["BTC"], 60, 40, exclude_self="BTC")
# gold beta
cands["gold_beta_60d"] = rolling_beta(rets, rets["XAU"], 60, 40, exclude_self="XAU")
# market correlation (systematic-ness)
cands["mkt_corr_60d"] = rets.rolling(60).corr(mkt)
# relative momentum 120d (vs cross-sectional median)
mom120 = per_asset(lambda s: s.pct_change().rolling(120).sum())
cands["rel_mom_120d"] = mom120 - mom120.median(axis=1)
# worst-day over 20d
cands["max_loss_20d"] = per_asset(lambda s: s.pct_change().rolling(20).min())
# hilo range
cands["hilo_range_20d"] = per_asset(lambda s: ((s.rolling(2).max() - s.rolling(2).min()) / s).rolling(20).mean())
# short-term reversal (5d)
cands["reversal_5d"] = -per_asset(lambda s: s.pct_change().rolling(5).sum())
# rsi14 (overbought/oversold)
cands["rsi_14d"] = per_asset(lambda s: rsi(s, 14))

fwd = forward_returns(closes, H_ADM)
print(f"{'factor':<24}{'n':>6}{'IC':>9}{'ICIR':>8}{'hit':>6}{'covA':>7}{'covD':>7}{'turn':>7}{'libcorr':>9}  decay10  maxcorr")
print("-" * 115)
t0 = time.time()
results = {}
for name, panel in cands.items():
    panel = panel.reindex(closes.index)
    ics = rank_ic_series_vec(panel, fwd, 8)
    if len(ics) < 200:
        print(f"{name:<24}{len(ics):>6}  too few dates")
        continue
    m = summarize_ic_vec(ics, 1)
    cov = coverage_metrics(panel)
    to = turnover_rank(panel, 10)
    dec = {}
    for h in HORIZONS:
        dh = rank_ic_series_vec(panel, forward_returns(closes, h), 8)
        dec[h] = round(float(dh.mean()), 4) if len(dh) else float("nan")
    corr, key = max_library_corr(panel, lib)
    flag = "PASS" if abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084 else ""
    print(f"{name:<24}{m['n_ic_dates']:>6}{m['ic']:>9.4f}{m['icir']:>8.4f}{m['ic_hit_ratio']:>6.3f}"
          f"{cov['coverage_asset_days']:>7.3f}{cov['coverage_dates_ge8']:>7.3f}{to:>7.3f}{corr:>9.3f}"
          f"  {dec[10]:.4f}  {key} {flag}")
    results[name] = (m, cov, to, dec, corr, key, ics, panel)
print(f"\nelapsed {time.time()-t0:.1f}s")
summary = {k: {"metrics": v[0], "coverage": v[1], "turnover": v[2], "decay": v[3],
               "libcorr": v[4], "maxcorr_factor": v[5]} for k, v in results.items()}
with open("scripts/_miner3_batchB_results.json", "w") as fh:
    json.dump(summary, fh, indent=1, default=str)
print("saved scripts/_miner3_batchB_results.json")
