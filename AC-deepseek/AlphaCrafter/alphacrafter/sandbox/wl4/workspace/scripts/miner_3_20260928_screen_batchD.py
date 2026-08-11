"""miner_3 batch D screen (2026-09-28, visible data through 2026-09-25).

Parts:
  A) Drift re-validation of 4 active library factors (h=10 gate).
  B) Re-run batch C candidates (macro-driver betas + price-shape) - results were lost.
  C) New batch D candidates:
       - relative/idiosyncratic momentum (vs equal-weight market)
       - realized skewness 20d
       - max drawdown 60d
       - vol-adjusted momentum (Sharpe-like 20d mom / 20d vol)
       - up/down market beta capture ratio 60d
       - 5d return autocorrelation
       - trend consistency (up-day fraction 20d)
       - OBV slope 20d
       - vol ratio 5x60 (vol regime expansion)
       - range position 20d (close within recent high-low band)
       - risk-adjusted 60d momentum (60d ret / 60d vol)
       - downside vol 20d

Admission gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10 (15-asset universe).
Also report max_abs_library_correlation vs all persisted library signals.
"""
import sys, time, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 coverage_metrics, turnover_rank, max_library_corr,
                                 rank_ic_series, summarize_ic)

panels = load_panels(days=3000)
closes = close_panel(panels)
rets = closes.pct_change()
H_ADM = 10
HORIZONS = (1, 2, 3, 5, 10, 20)
MIN_VALID = 8

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

# ---------------- library signals (all persisted/evicted factors) ----------------
lib = {}
mkt = rets.mean(axis=1)
dn = mkt.where(mkt < 0).fillna(0.0)
up = mkt.where(mkt > 0).fillna(0.0)
lib["mom_10d_skip5"] = closes.shift(5) / closes.shift(15) - 1.0
lib["mom_120d_skip5"] = closes.shift(5) / closes.shift(125) - 1.0
lib["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
lib["dn_mkt_beta_60d"] = rolling_beta(rets, dn, 60, 40)
vix = panels["VIX"]["close"].astype(float)
vix_ret = vix.pct_change()
lib["vix_beta_cond_60x20"] = -rolling_beta(rets, vix_ret, 60, 40) * (vix / vix.shift(20) - 1.0)
lib["eurusd_beta_60d"] = rolling_beta(rets, panels["EURUSD"]["close"].astype(float).pct_change(), 60, 40)
lib["rate_beta_cn10y_60d"] = rolling_beta(rets, panels["CN10Y"]["close"].astype(float).pct_change(), 60, 40)
vol_panel = pd.concat({a: panels[a]["volume"].astype(float) for a in closes.columns}, axis=1).reindex(closes.index)
def roll_corr(a, b, win=20):
    z = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return z["a"].rolling(win).corr(z["b"])
vpc = {a: roll_corr(rets[a], vol_panel[a]) for a in closes.columns}
lib["vol_price_corr_20"] = pd.DataFrame(vpc, index=rets.index)
for k in lib:
    lib[k] = lib[k].reindex(closes.index)

# ---------------- vectorized rank IC ----------------
def rank_ic_series_vec(factor_panel, fwd, min_valid=MIN_VALID):
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

def evaluate(name, panel, es=1):
    panel = panel.reindex(closes.index)
    ics = rank_ic_series_vec(panel, fwd10, MIN_VALID)
    if len(ics) < 200:
        print(f"{name:<26}{len(ics):>6}  too few dates")
        return None
    m = summarize_ic_vec(ics, es)
    cov = coverage_metrics(panel)
    to = turnover_rank(panel, 10)
    dec = {}
    for h in HORIZONS:
        dh = rank_ic_series_vec(panel, forward_returns(closes, h), MIN_VALID)
        dec[h] = round(float(dh.mean()), 4) if len(dh) else float("nan")
    corr, key = max_library_corr(panel, lib)
    flag = "PASS" if abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084 else ""
    print(f"{name:<26}{m['n_ic_dates']:>6}{m['ic']:>9.4f}{m['icir']:>8.4f}{m['ic_hit_ratio']:>6.3f}"
          f"{cov['coverage_asset_days']:>7.3f}{cov['coverage_dates_ge8']:>7.3f}{to:>7.3f}{corr:>9.3f}"
          f"  {dec[1]:.4f} {dec[3]:.4f} {dec[10]:.4f} {dec[20]:.4f}  {key} {flag}")
    return {"metrics": m, "coverage": cov, "turnover": to, "decay": dec,
            "libcorr": corr, "maxcorr_factor": key, "n_ic_dates": m["n_ic_dates"]}

fwd10 = forward_returns(closes, H_ADM)
print(f"panels: {len(closes)} dates {closes.index[0].date()}..{closes.index[-1].date()} | assets={closes.shape[1]}")
print(f"{'factor':<26}{'n':>6}{'IC':>9}{'ICIR':>8}{'hit':>6}{'covA':>7}{'covD':>7}{'turn':>7}{'libcorr':>9}  dec1 dec3 dec10 dec20  maxcorr")
print("-" * 122)
results = {}

# ---- A) active library drift re-validation ----
print("\n=== A. ACTIVE LIBRARY RE-VALIDATION (drift check, h=10 gate) ===")
for name in ["dn_mkt_beta_60d", "eurusd_beta_60d", "rate_beta_cn10y_60d", "vol_price_corr_20"]:
    es = 1 if name in ("dn_mkt_beta_60d", "vol_price_corr_20") else -1
    results["active_" + name] = evaluate("ACTIVE " + name, lib[name], es)

# ---- B) batch C candidates (rerun) ----
print("\n=== B. BATCH C CANDIDATES ===")
cands = {}
dxy = panels["DXY"]["close"].astype(float)
usdjpy = panels["USDJPY"]["close"].astype(float)
us10 = panels["US10Y"]["close"].astype(float)
cn10 = panels["CN10Y"]["close"].astype(float)
spread = (us10 - cn10)
cands["dxy_beta_60d"] = rolling_beta(rets, dxy.pct_change(), 60, 40)
cands["usdjpy_beta_60d"] = rolling_beta(rets, usdjpy.pct_change(), 60, 40)
cands["wti_beta_60d"] = rolling_beta(rets, rets["WTI"], 60, 40, exclude_self="WTI")
cands["copper_beta_60d"] = rolling_beta(rets, rets["COPPER"], 60, 40, exclude_self="COPPER")
cands["us10y_beta_60d"] = rolling_beta(rets, rets["US10Y"], 60, 40, exclude_self="US10Y")
cands["spread_beta_60d"] = rolling_beta(rets, spread.pct_change(), 60, 40)
cands["vix_beta_60d"] = rolling_beta(rets, vix_ret, 60, 40)
cands["mkt_beta_60d"] = rolling_beta(rets, mkt, 60, 40)
cands["zscore_20d"] = per_asset(lambda s: (s - s.rolling(20).mean()) / s.rolling(20).std())
def semi_dev_ratio(s, win=20):
    r = s.pct_change()
    neg = r.where(r < r.rolling(win).mean())
    down = (neg - neg.rolling(win).mean()).pow(2).rolling(win).mean().pow(0.5)
    tot = r.rolling(win).std()
    return down / tot
cands["semi_dev_ratio_20d"] = per_asset(lambda s: semi_dev_ratio(s, 20))
amihud = (rets.abs() / vol_panel.replace(0, np.nan)).rolling(20).mean()
cands["amihud_20d"] = amihud
def eff_ratio(s, win=10):
    d = s.diff().abs().rolling(win).sum()
    return (s - s.shift(win)).abs() / d
cands["eff_ratio_10d"] = per_asset(lambda s: eff_ratio(s, 10))
vol20 = rets.rolling(20).std()
cands["vol_pctile_20x250"] = vol20.rolling(250).apply(lambda x: (x.iloc[-1] >= x).mean() if len(x) >= 120 else np.nan, raw=False)
def range_vol(s, win=20):
    hi = s.rolling(2).max(); lo = s.rolling(2).min()
    rng = ((hi - lo) / s).rolling(win).mean()
    return rng / s.pct_change().rolling(win).std()
cands["range_vol_ratio_20d"] = per_asset(lambda s: range_vol(s, 20))
cands["gain_loss_asym_20d"] = per_asset(lambda s: s.pct_change().rolling(20).max() + s.pct_change().rolling(20).min())
for name, panel in cands.items():
    results["B_" + name] = evaluate("B " + name, panel, 1)

# ---- C) new batch D candidates ----
print("\n=== C. BATCH D NEW CANDIDATES ===")
candsD = {}
# C1: idiosyncratic (relative) momentum 60d vs equal-weight market
candsD["rel_mom_60d"] = (closes.shift(5) / closes.shift(65) - 1.0) - (mkt.shift(5) / mkt.shift(65) - 1.0)
# C2: realized skewness 20d
candsD["skew_20d"] = rets.rolling(20).skew()
# C3: max drawdown 60d (negative = drawdown)
candsD["drawdown_60d"] = closes / closes.rolling(60).max() - 1.0
# C4: vol-adjusted momentum 20d (mom skip1 / vol20)
candsD["vol_adj_mom_20x20"] = (closes / closes.shift(20) - 1.0) / vol20
# C5: up/down beta capture ratio 60d
up_beta = rolling_beta(rets, up, 60, 40)
dn_beta = rolling_beta(rets, dn, 60, 40)
candsD["capture_ratio_60d"] = up_beta / dn_beta.replace(0, np.nan)
# C6: 5d return autocorrelation
def autocorr5(s):
    r = s.pct_change()
    return r.rolling(20).apply(lambda x: np.corrcoef(x[:-5], x[5:])[0, 1] if len(x) >= 12 else np.nan, raw=True)
candsD["autocorr_5d_20w"] = per_asset(autocorr5)
# C7: trend consistency 20d (up-day fraction)
candsD["trend_consistency_20d"] = (rets > 0).rolling(20).mean()
# C8: OBV slope 20d
obv = (np.sign(rets) * vol_panel).fillna(0.0).cumsum()
candsD["obv_slope_20d"] = obv - obv.shift(20)
# C9: vol ratio 5x60 (short-term vol expansion)
candsD["vol_ratio_5x60"] = rets.rolling(5).std() / rets.rolling(60).std()
# C10: range position 20d
def range_pos(s, win=20):
    hi = s.rolling(win).max(); lo = s.rolling(win).min()
    return (s - lo) / (hi - lo).replace(0, np.nan)
candsD["range_pos_20d"] = per_asset(lambda s: range_pos(s, 20))
# C11: risk-adjusted 60d momentum
vol60 = rets.rolling(60).std()
candsD["mom60_vol60"] = (closes.shift(5) / closes.shift(65) - 1.0) / vol60
# C12: downside vol 20d
candsD["downside_vol_20d"] = rets.where(rets < 0, 0.0).rolling(20).std()
for name, panel in candsD.items():
    results["D_" + name] = evaluate("D " + name, panel, 1)

print(f"\nelapsed {time.time()-0:.1f}s")
with open("scripts/_miner3_batchD_results.json", "w") as fh:
    json.dump(results, fh, indent=1, default=str)
print("saved scripts/_miner3_batchD_results.json")
