"""miner_3 batch C screen (2026-09-14): macro-driver beta factors + price-shape/quality factors.

Theme 1 (macro-beta family): rolling 60d beta of each asset's return to macro/commodity
drivers - DXY, USDJPY, WTI, COPPER, US10Y, US-CN yield spread change, plain VIX.
Theme 2 (price-shape family): z-score (bollinger) position, downside-vol ratio,
Amihud illiquidity, Kaufman efficiency ratio, vol percentile, range/vol ratio,
max-gain-minus-loss asymmetry.

Admission gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10 (15-asset universe).
Library correlation vs all persisted factor signals (4 effective + evicted).
Data through visible date 2026-09-11 (no lookahead).
"""
import sys, time, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 coverage_metrics, turnover_rank, max_library_corr)

panels = load_panels(days=3000)
closes = close_panel(panels)
rets = closes.pct_change()
H_ADM = 10
HORIZONS = (1, 2, 3, 5, 10, 20)

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

def per_asset(func):
    out = {}
    for a in closes.columns:
        s = closes[a].dropna()
        out[a] = func(s)
    return pd.DataFrame(out).reindex(closes.index)

# ---------------- recompute ALL library factor signals ----------------
def rolling_beta_lib(asset_ret, driver_ret, win=60, min_obs=40, exclude_self=None):
    return rolling_beta(asset_ret, driver_ret, win, min_obs, exclude_self)

lib = {}
mkt = rets.mean(axis=1)
dn = mkt.where(mkt < 0).fillna(0.0)
lib["mom_10d_skip5"] = closes.shift(5) / closes.shift(15) - 1.0
lib["mom_120d_skip5"] = closes.shift(5) / closes.shift(125) - 1.0
lib["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
lib["dn_mkt_beta_60d"] = rolling_beta_lib(rets, dn, 60, 40)
vix = panels["VIX"]["close"].astype(float)
vix_ret = vix.pct_change()
lib["vix_beta_cond_60x20"] = -rolling_beta_lib(rets, vix_ret, 60, 40) * (vix / vix.shift(20) - 1.0)
lib["eurusd_beta_60d"] = rolling_beta_lib(rets, panels["EURUSD"]["close"].astype(float).pct_change(), 60, 40)
lib["rate_beta_cn10y_60d"] = rolling_beta_lib(rets, panels["CN10Y"]["close"].astype(float).pct_change(), 60, 40)
# vol_price_corr_20: rolling corr(ret, volume, 20)
vol_panel = pd.concat({a: panels[a]["volume"].astype(float) for a in closes.columns}, axis=1).reindex(closes.index)
def roll_corr(a, b, win=20):
    z = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return z["a"].rolling(win).corr(z["b"])
vpc = {a: roll_corr(rets[a], vol_panel[a]) for a in closes.columns}
lib["vol_price_corr_20"] = pd.DataFrame(vpc, index=rets.index)
for k in lib:
    lib[k] = lib[k].reindex(closes.index)

# ---------------- vectorized rank IC ----------------
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

# ---------------- candidate factors ----------------
cands = {}
# Theme 1: macro-driver betas (60d)
dxy = panels["DXY"]["close"].astype(float)
cands["dxy_beta_60d"] = rolling_beta(rets, dxy.pct_change(), 60, 40)
usdjpy = panels["USDJPY"]["close"].astype(float)
cands["usdjpy_beta_60d"] = rolling_beta(rets, usdjpy.pct_change(), 60, 40)
cands["wti_beta_60d"] = rolling_beta(rets, rets["WTI"], 60, 40, exclude_self="WTI")
cands["copper_beta_60d"] = rolling_beta(rets, rets["COPPER"], 60, 40, exclude_self="COPPER")
cands["us10y_beta_60d"] = rolling_beta(rets, rets["US10Y"], 60, 40, exclude_self="US10Y")
# yield spread (US10Y - CN10Y) change beta
us10 = panels["US10Y"]["close"].astype(float)
cn10 = panels["CN10Y"]["close"].astype(float)
spread = (us10 - cn10)
cands["spread_beta_60d"] = rolling_beta(rets, spread.pct_change(), 60, 40)
# plain VIX beta (unconditional)
cands["vix_beta_60d"] = rolling_beta(rets, vix_ret, 60, 40)
# full market beta (not downside-conditioned)
cands["mkt_beta_60d"] = rolling_beta(rets, mkt, 60, 40)

# Theme 2: price-shape / quality
# z-score / bollinger position 20d
cands["zscore_20d"] = per_asset(lambda s: (s - s.rolling(20).mean()) / s.rolling(20).std())
# downside-vol ratio 20d: std of negative deviations / total std
def semi_dev_ratio(s, win=20):
    r = s.pct_change()
    neg = r.where(r < r.rolling(win).mean())
    down = (neg - neg.rolling(win).mean()).pow(2).rolling(win).mean().pow(0.5)
    tot = r.rolling(win).std()
    return down / tot
cands["semi_dev_ratio_20d"] = per_asset(lambda s: semi_dev_ratio(s, 20))
# Amihud illiquidity 20d: mean(|ret|/volume)
amihud = (rets.abs() / vol_panel.replace(0, np.nan)).rolling(20).mean()
cands["amihud_20d"] = amihud
# Kaufman efficiency ratio 10d: |close - close.shift(10)| / sum(|diff|,10)
def eff_ratio(s, win=10):
    d = s.diff().abs().rolling(win).sum()
    return (s - s.shift(win)).abs() / d
cands["eff_ratio_10d"] = per_asset(lambda s: eff_ratio(s, 10))
# vol percentile 20x250: rank of 20d vol within trailing 250d
vol20 = rets.rolling(20).std()
cands["vol_pctile_20x250"] = vol20.rolling(250).apply(lambda x: (x.iloc[-1] >= x).mean() if len(x) >= 120 else np.nan, raw=False)
# range/vol ratio 20d: mean((high-low)/close)/vol20
def range_vol(s, win=20):
    hi = s.rolling(2).max(); lo = s.rolling(2).min()
    rng = ((hi - lo) / s).rolling(win).mean()
    return rng / s.pct_change().rolling(win).std()
cands["range_vol_ratio_20d"] = per_asset(lambda s: range_vol(s, 20))
# max gain + max loss 20d asymmetry
cands["gain_loss_asym_20d"] = per_asset(lambda s: s.pct_change().rolling(20).max() + s.pct_change().rolling(20).min())

fwd = forward_returns(closes, H_ADM)
print(f"date range: {closes.index[0].date()} .. {closes.index[-1].date()}  n_dates={len(closes)}  n_assets={closes.shape[1]}")
print(f"{'factor':<24}{'n':>6}{'IC':>9}{'ICIR':>8}{'hit':>6}{'covA':>7}{'covD':>7}{'turn':>7}{'libcorr':>9}  dec1 dec3 dec10 dec20  maxcorr")
print("-" * 120)
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
          f"  {dec[1]:.4f} {dec[3]:.4f} {dec[10]:.4f} {dec[20]:.4f}  {key} {flag}")
    results[name] = {"metrics": m, "coverage": cov, "turnover": to, "decay": dec,
                     "libcorr": corr, "maxcorr_factor": key, "n_ic_dates": m["n_ic_dates"]}
print(f"\nelapsed {time.time()-t0:.1f}s  candidates={len(cands)}")
with open("scripts/_miner3_batchC_results.json", "w") as fh:
    json.dump(results, fh, indent=1, default=str)
print("saved scripts/_miner3_batchC_results.json")
