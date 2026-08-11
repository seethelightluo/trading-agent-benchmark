"""miner_1 2026-11-09: batch F - re-validation + fresh candidates.

Context: online trading since 2026-07-16; current date 2026-11-09, data visible
through 2026-11-06. Library holds 4 effective factors:
  vol_price_corr_20 (+), eurusd_beta_60d (-), rate_beta_cn10y_60d (-),
  dn_mkt_beta_60d (+).

Contents:
  0. Probe data window.
  1. Active library re-validation (drift check on extended data).
  2. Re-check previously passing-but-unpersisted candidates (skew_20d,
     max_ret_20d) with current data + rho vs CURRENT 4-factor library.
  3. New candidates: abnormal volume, up-volume ratio, risk-adjusted momentum,
     macro betas (XAU/WTI/US10Y/DXY/USDJPY), tail risk (CVaR), return
     autocorrelation, Kaufman efficiency 60, downside-vol ratio, gap mean.

Admission gate (shared 15-asset universe): |IC| >= 0.0070 AND |ICIR| >= 0.0840
at h=10, n_ic_dates >= 200, and max_abs_library_correlation < 0.5 vs the
current effective library. No lookahead: factor at t uses data <= t; forward
ret = close[t+h]/close[t]-1.
"""
import sys, time, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, ret_panel,
                                 forward_returns, rank_ic_series, summarize_ic,
                                 coverage_metrics, turnover_rank, max_library_corr,
                                 decay_profile)

t0 = time.time()
panels = load_panels()
closes = close_panel(panels)
rets = ret_panel(panels)
H_ADM = 10
HORIZONS = (1, 2, 3, 5, 10, 20)
MIN_VALID = 8
MIN_IC_DATES = 200
print(f"panels={len(panels)} closes={closes.shape} dates="
      f"{closes.index[0].date()}..{closes.index[-1].date()} load={time.time()-t0:.1f}s", flush=True)


# ---------------- vectorized rank-IC (verified against loop in prior cycles) ----------------
def rank_ic_series_vec(factor_panel: pd.DataFrame, fwd: pd.DataFrame, min_valid: int = 8):
    fr = factor_panel.rank(axis=1, method="average", na_option="keep")
    rr = fwd.rank(axis=1, method="average", na_option="keep")
    valid = fr.notna() & rr.notna()
    n = valid.sum(axis=1).astype(float)
    f = fr.where(valid)
    r = rr.where(valid)
    fm = f.sub(f.mean(axis=1), axis=0)
    rm = r.sub(r.mean(axis=1), axis=0)
    num = fm.mul(rm).sum(axis=1, min_count=1)
    den = np.sqrt(fm.pow(2).sum(axis=1, min_count=1) * rm.pow(2).sum(axis=1, min_count=1))
    ic = num.div(den.replace(0, np.nan))
    ic = ic.where((n >= min_valid) & (n >= 2) & (den > 1e-14))
    ic = ic.replace([np.inf, -np.inf], np.nan).dropna()
    ic.name = "ic"
    return ic


def summarize(ic_series, expected_sign=1):
    ic = float(ic_series.mean())
    std = float(ic_series.std(ddof=1))
    icir = ic / std if std > 0 else 0.0
    hit = float((np.sign(ic_series) == expected_sign).mean())
    return {"ic": round(ic, 4), "icir": round(icir, 4), "ic_hit_ratio": round(hit, 3),
            "n_ic_dates": int(len(ic_series)), "ic_std": round(std, 4)}


def evaluate(name, panel, exp_sign, lib, verbose=True):
    panel = panel.reindex(closes.index)
    fwd = forward_returns(closes, H_ADM)
    ics = rank_ic_series_vec(panel, fwd, MIN_VALID)
    m = summarize(ics, exp_sign)
    m.update(coverage_metrics(panel, min_valid=MIN_VALID))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    m["decay_ic_by_horizon"] = decay_profile(panel, closes, HORIZONS, MIN_VALID, exp_sign)
    corr, key = max_library_corr(panel, lib)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    gate = (abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084
            and m["n_ic_dates"] >= MIN_IC_DATES and corr < 0.5)
    if verbose:
        print(f"{name:20s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
              f"n={m['n_ic_dates']:5d} covAD={m['coverage_asset_days']:.3f} covD8={m['coverage_dates_ge8']:.3f} "
              f"to={m.get('turnover_10d_rank')} rho={corr:.3f}({key}) "
              f"decay={ {k: round(v,3) for k,v in m['decay_ic_by_horizon'].items()} } "
              f"{'-> PASS' if gate else ''}", flush=True)
    return m, ics


def per_asset(build, min_len=260):
    cols = {}
    for a in closes.columns:
        s = closes[a].dropna()
        if len(s) < min_len:
            continue
        f = build(s, a)
        if f is None or len(f) == 0:
            continue
        if isinstance(f, pd.Series):
            f = f.reindex(s.index)
        else:
            f = pd.Series(np.asarray(f), index=s.index[:len(f)])
        cols[a] = f
    return pd.DataFrame(cols).reindex(closes.index)


def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40, exclude_self=None):
    beta = {}
    for a in asset_ret.columns:
        if exclude_self is not None and a == exclude_self:
            beta[a] = pd.Series(np.nan, index=asset_ret.index)
            continue
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        b = z["a"].rolling(win).cov(z["m"]) / z["m"].rolling(win).var()
        beta[a] = b.where(z["m"].rolling(win).count() >= min_obs)
    return pd.DataFrame(beta, index=asset_ret.index)


# ---------------- current library signals (4 effective factors) ----------------
mkt = rets.mean(axis=1)
dn = mkt.where(mkt < 0).fillna(0.0)
vix = panels["VIX"]["close"].astype(float)
eur = panels["EURUSD"]["close"].astype(float)
cn10 = panels["CN10Y"]["close"].astype(float)


def vpc_build(s, a):
    r = s.pct_change()
    v = panels[a]["volume"].astype(float).reindex(s.index)
    z = pd.concat([r.rename("a"), v.rename("v")], axis=1).dropna()
    return z["a"].rolling(20, min_periods=10).corr(z["v"])


lib = {
    "vol_price_corr_20": per_asset(vpc_build),
    "eurusd_beta_60d": rolling_beta(rets, eur.pct_change(), 60),
    "rate_beta_cn10y_60d": rolling_beta(rets, cn10.pct_change(), 60),
    "dn_mkt_beta_60d": rolling_beta(rets, dn, 60),
}
for k in lib:
    lib[k] = lib[k].reindex(closes.index)

# ============ 0. PROBE ============
print("\n=== 0. WINDOW ===", flush=True)
print(f"dates={closes.index[0].date()}..{closes.index[-1].date()} n_dates={len(closes)} "
      f"n_assets={closes.shape[1]}", flush=True)
print("assets with >=2000 obs:", [a for a in closes.columns if closes[a].notna().sum() >= 2000], flush=True)

# ============ 1. ACTIVE LIBRARY DRIFT ============
print("\n=== 1. ACTIVE LIBRARY RE-VALIDATION (drift, extended window) ===", flush=True)
active = {
    "vol_price_corr_20": (lib["vol_price_corr_20"], 1),
    "eurusd_beta_60d": (lib["eurusd_beta_60d"], -1),
    "rate_beta_cn10y_60d": (lib["rate_beta_cn10y_60d"], -1),
    "dn_mkt_beta_60d": (lib["dn_mkt_beta_60d"], 1),
}
active_meta = {}
for name, (panel, es) in active.items():
    m, ics = evaluate(f"[active]{name}", panel, es, lib)
    active_meta[name] = m

# ============ 2. PREVIOUSLY PASSING CANDIDATES ============
print("\n=== 2. RE-CHECK PREVIOUSLY PROMISING (batchC2, unpersisted) ===", flush=True)
recheck = {}

def f_sk(s, a):
    return s.pct_change().rolling(20).skew()
recheck["skew_20d"] = (per_asset(f_sk), -1)

def f_mx(s, a):
    return s.pct_change().rolling(20).max()
recheck["max_ret_20d"] = (per_asset(f_mx), -1)

recheck_meta = {}
for name, (panel, es) in recheck.items():
    m, ics = evaluate(name, panel, es, lib)
    recheck_meta[name] = m

# ============ 3. NEW CANDIDATES ============
print("\n=== 3. NEW CANDIDATES (batch F) ===", flush=True)
new = {}

# 3.1 abnormal volume: 20d avg volume / 60d avg volume - 1 (volume surge)
def f_vz(s, a):
    v = panels[a]["volume"].astype(float).reindex(s.index)
    return v.rolling(20).mean() / v.rolling(60).mean() - 1.0
new["vol_surge_20_60"] = (per_asset(f_vz), -1)

# 3.2 up-volume ratio: share of volume on up days over 20d
def f_uvr(s, a):
    r = s.pct_change()
    v = panels[a]["volume"].astype(float).reindex(s.index)
    z = pd.concat([r.rename("r"), v.rename("v")], axis=1).dropna()
    up = z["v"].where(z["r"] > 0).rolling(20).sum()
    tot = z["v"].rolling(20).sum()
    return (up / tot.replace(0, np.nan)).reindex(s.index)
new["up_vol_ratio_20"] = (per_asset(f_uvr), 1)

# 3.3 risk-adjusted momentum: 60d return / 20d vol
def f_mrv(s, a):
    r = s.pct_change()
    return (s / s.shift(60) - 1.0) / r.rolling(20).std()
new["mom_risky_60_20"] = (per_asset(f_mrv), 1)

# 3.4 macro betas
new["xau_beta_60d"] = (rolling_beta(rets, rets["XAU"], 60), 1)          # safe-haven beta
new["wti_beta_60d"] = (rolling_beta(rets, rets["WTI"], 60), 1)          # energy beta
new["us10y_beta_60d"] = (rolling_beta(rets, rets["US10Y"], 60), -1)     # US rate sensitivity
new["usdjpy_beta_60d"] = (rolling_beta(rets, panels["USDJPY"]["close"].astype(float).pct_change(), 60), -1)
new["dxy_beta_60d"] = (rolling_beta(rets, panels["DXY"]["close"].astype(float).pct_change(), 60), 1)

# 3.5 tail risk: mean of worst 5% daily returns over 60d (CVaR proxy)
def f_cvar(s, a):
    r = s.pct_change()
    return r.rolling(60).apply(lambda x: np.nanmean(np.sort(x)[:max(1, int(0.05 * len(x)))]), raw=True)
new["cvar_60"] = (per_asset(f_cvar), -1)

# 3.6 return autocorrelation 5d (trend persistence)
def f_ac(s, a):
    r = s.pct_change()
    return r.rolling(10).apply(lambda x: pd.Series(x).autocorr(1) if len(x) > 3 else np.nan, raw=True)
new["autocorr_10"] = (per_asset(f_ac), 1)

# 3.7 Kaufman efficiency 60
def f_ke60(s, a):
    r = s.pct_change()
    return (s - s.shift(60)).abs() / r.abs().rolling(60).sum()
new["kaufman_eff_60"] = (per_asset(f_ke60), 1)

# 3.8 downside vol ratio: downside deviation / total vol (20d)
def f_dvr(s, a):
    r = s.pct_change()
    dd = r.clip(upper=0.0).rolling(20).std()
    return dd / r.rolling(20).std()
new["downside_vol_ratio_20"] = (per_asset(f_dvr), -1)

# 3.9 mean overnight-style gap: open/prev_close - 1 over 20d (gap persistence)
def f_gap(s, a):
    o = panels[a]["open"].astype(float).reindex(s.index)
    g = o / s.shift(1) - 1.0
    return g.rolling(20).mean()
new["gap_mean_20"] = (per_asset(f_gap), 1)

new_meta = {}
for name, (panel, es) in new.items():
    m, ics = evaluate(name, panel, es, lib)
    new_meta[name] = m

print(f"\nDone. elapsed={time.time()-t0:.1f}s", flush=True)

out = {"active": active_meta, "recheck": recheck_meta, "new": new_meta,
       "window": {"first": str(closes.index[0].date()), "last": str(closes.index[-1].date()),
                  "n_dates": int(len(closes)), "n_assets": int(closes.shape[1])}}
with open("scripts/_miner1_20261109_results.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print("saved scripts/_miner1_20261109_results.json")
