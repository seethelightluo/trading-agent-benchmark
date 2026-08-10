"""miner_3 batch A screen (vectorized): per-asset factor computation + vectorized rank IC.

Admission gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10.
"""
import sys, time
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 coverage_metrics, turnover_rank, library_signals,
                                 max_library_corr)

panels = load_panels(days=3000)
closes = close_panel(panels)
rets = closes.pct_change()
lib = library_signals(panels, closes, rets)
H_ADM = 10
HORIZONS = (1, 3, 5, 10, 20)

def rank_ic_series_vec(factor_panel, fwd, min_valid=8):
    """Vectorized daily Spearman rank IC (Pearson on ranks)."""
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

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0.0).rolling(n).mean()
    dn = (-d.clip(upper=0.0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)

cands = {}
cands["range_pos_20d"] = per_asset(lambda s: (s - s.rolling(20).min()) / (s.rolling(20).max() - s.rolling(20).min()))
cands["range_pos_60d"] = per_asset(lambda s: (s - s.rolling(60).min()) / (s.rolling(60).max() - s.rolling(60).min()))
cands["rsi_14"] = per_asset(lambda s: rsi(s, 14))
cands["drawdown_60d"] = per_asset(lambda s: s / s.rolling(60).max() - 1.0)
cands["skew_60d"] = per_asset(lambda s: s.pct_change().rolling(60).skew())
cands["downside_semidev_60d"] = per_asset(lambda s: np.sqrt((s.pct_change().clip(upper=0.0) ** 2).rolling(60).mean()))
cands["vol_ratio_5x60"] = per_asset(lambda s: s.pct_change().rolling(5).std() / s.pct_change().rolling(60).std())
cands["eff_ratio_20d"] = per_asset(lambda s: (s - s.shift(20)).abs() / s.pct_change().abs().rolling(20).sum())
cands["mom_quality_20d"] = per_asset(lambda s: s.pct_change().rolling(20).sum() / s.pct_change().rolling(20).std())
cands["macd_12x26"] = per_asset(lambda s: (s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()) / s)

mom20 = per_asset(lambda s: s.pct_change().rolling(20).sum())
cands["rel_strength_20d"] = mom20 - mom20.median(axis=1)

dxy = panels["DXY"]["close"].astype(float)
cands["dxy_beta_60d"] = rolling_beta(rets, dxy.pct_change(), 60, 40)
xau = closes["XAU"]
cands["xau_beta_60d"] = rolling_beta(rets, xau.pct_change(), 60, 40, exclude_self="XAU")
btc = closes["BTC"]
cands["btc_beta_60d"] = rolling_beta(rets, btc.pct_change(), 60, 40, exclude_self="BTC")
mkt = rets.mean(axis=1)
up = mkt.where(mkt > 0).fillna(0.0)
dn = mkt.where(mkt < 0).fillna(0.0)
cands["updn_beta_diff_60d"] = rolling_beta(rets, up, 60, 40) - rolling_beta(rets, dn, 60, 40)

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
# persist intermediate metrics for later persistence step
import json
summary = {k: {"metrics": v[0], "coverage": v[1], "turnover": v[2], "decay": v[3],
               "libcorr": v[4], "maxcorr_factor": v[5]} for k, v in results.items()}
with open("scripts/_miner3_batchA_results.json", "w") as fh:
    json.dump(summary, fh, indent=1, default=str)
print("saved scripts/_miner3_batchA_results.json")
