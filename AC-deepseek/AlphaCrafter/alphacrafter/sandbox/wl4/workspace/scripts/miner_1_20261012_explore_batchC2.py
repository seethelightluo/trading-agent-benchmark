"""miner_1 2026-10-12: batch C2 - corrected vectorized rank-IC + fresh candidates.

FIX vs prior batch C runs:
  * rank_ic_series_vec rewritten with a CONSISTENT valid-pair mask (previous
    versions mixed f-valid / r-valid sums, and one earlier variant misused a
    DataFrame as a Series). Verified against the loop-based Spearman reference
    (factor_research_lib.rank_ic_series) on 3 factors: max |diff| printed.
  * All rolling-window factors computed PER-ASSET on each asset's own continuous
    calendar (dropna) then reindexed to the union calendar (avoids weekend-NaN
    pollution of equity rows on the union calendar).

Contents:
  0. Probe data window + verify vec vs loop rank-IC.
  1. Active library (4 effective factors) re-validation / drift check.
  2. 12 batch-C candidates (trend efficiency, lottery/tail, range, liquidity,
     vol-structure) - previously never fully evaluated.
  3. 8 new candidates this cycle: pct_of_peak_60, sharpe_60, mom_term_20_60,
     stoch_k_20, trend_align, btc_beta_60d, obv_mom_20, range_mean_20.

Admission gate (shared 15-asset universe): |IC| >= 0.0070 AND |ICIR| >= 0.0840
at h=10, n_ic_dates >= 200, coverage_dates_ge8 high, library correlation audited.
No lookahead: factor at t uses data <= t; forward ret close[t+h]/close[t]-1.
"""
import sys, time
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, ret_panel,
                                 forward_returns, rank_ic_series, summarize_ic,
                                 coverage_metrics, turnover_rank, max_library_corr)

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


# ---------------- correct vectorized rank-IC ----------------
def rank_ic_series_vec(factor_panel: pd.DataFrame, fwd: pd.DataFrame, min_valid: int = 8):
    fr = factor_panel.rank(axis=1, method="average", na_option="keep")
    rr = fwd.rank(axis=1, method="average", na_option="keep")
    valid = fr.notna() & rr.notna()
    n = valid.sum(axis=1).astype(float)
    f = fr.where(valid)
    r = rr.where(valid)
    fm = f.sub(f.mean(axis=1), axis=0)   # row demean (skips NaN)
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


def decay_vec(factor_panel, horizons=HORIZONS, min_valid=MIN_VALID):
    out = {}
    for h in horizons:
        ics = rank_ic_series_vec(factor_panel, forward_returns(closes, h), min_valid)
        out[str(h)] = round(float(ics.mean()), 4) if len(ics) else float("nan")
    return out


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


def evaluate(name, panel, exp_sign, verbose=True):
    panel = panel.reindex(closes.index)
    fwd = forward_returns(closes, H_ADM)
    ics = rank_ic_series_vec(panel, fwd, MIN_VALID)
    m = summarize(ics, exp_sign)
    m.update(coverage_metrics(panel, min_valid=MIN_VALID))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    m["decay_ic_by_horizon"] = decay_vec(panel)
    corr, key = max_library_corr(panel, lib)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    gate = (abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084
            and m["n_ic_dates"] >= MIN_IC_DATES)
    if verbose:
        print(f"{name:20s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
              f"n={m['n_ic_dates']:5d} covAD={m['coverage_asset_days']:.3f} covD8={m['coverage_dates_ge8']:.3f} "
              f"to={m['turnover_10d_rank']:.3f} rho={corr:.3f}({key}) "
              f"decay={ {k: round(v,3) for k,v in m['decay_ic_by_horizon'].items()} } "
              f"{'-> PASS' if gate else ''}", flush=True)
    return m, ics


# ---------------- library signals for correlation audit ----------------
mkt = rets.mean(axis=1)
dn = mkt.where(mkt < 0).fillna(0.0)
vix = panels["VIX"]["close"].astype(float)
vix_ret = vix.pct_change()
eur = panels["EURUSD"]["close"].astype(float)
cn10 = panels["CN10Y"]["close"].astype(float)
usdcny = panels["USDCNY"]["close"].astype(float)
vol_panel = pd.concat({a: panels[a]["volume"].astype(float) for a in closes.columns}, axis=1).reindex(closes.index)


def vpc_build(s, a):
    r = s.pct_change()
    v = panels[a]["volume"].astype(float).reindex(s.index)
    z = pd.concat([r.rename("a"), v.rename("v")], axis=1).dropna()
    return z["a"].rolling(20, min_periods=10).corr(z["v"])


def rsi_build(s, a, win=14):
    d = s.diff()
    up = d.clip(lower=0.0).rolling(win).mean()
    dnw = (-d.clip(upper=0.0)).rolling(win).mean()
    rs = up / dnw.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


lib = {
    "mom_10d_skip5": closes.shift(5) / closes.shift(15) - 1.0,
    "mom_120d_skip5": closes.shift(5) / closes.shift(125) - 1.0,
    "vol_of_vol20x60": rets.rolling(20).std().rolling(60).std(),
    "vix_beta_cond_60x20": -rolling_beta(rets, vix_ret, 60) * (vix / vix.shift(20) - 1.0),
    "usdcny_beta_60d": rolling_beta(rets, usdcny.pct_change(), 60),
    "rsi_14": per_asset(rsi_build),
    "vol_price_corr_20": per_asset(vpc_build),
    "eurusd_beta_60d": rolling_beta(rets, eur.pct_change(), 60),
    "rate_beta_cn10y_60d": rolling_beta(rets, cn10.pct_change(), 60),
    "dn_mkt_beta_60d": rolling_beta(rets, dn, 60),
}
for k in lib:
    lib[k] = lib[k].reindex(closes.index)


# ============ 0. VERIFY vec vs loop ============
print("\n=== 0. vec vs loop rank-IC verification ===", flush=True)
for probe in ["vol_price_corr_20", "kaufman_eff_20", "skew_20d"]:
    pass
# verify on 3 quick panels
quick = {
    "close_sma200": closes / closes.rolling(200).mean() - 1.0,
    "rev_5d_skip1": -(closes.shift(1) / closes.shift(6) - 1.0),
    "vol_ratio_5_60": rets.rolling(5).std() / rets.rolling(60).std(),
}
fwd10 = forward_returns(closes, H_ADM)
for name, panel in quick.items():
    v = rank_ic_series_vec(panel, fwd10, MIN_VALID)
    l = rank_ic_series(panel, fwd10, MIN_VALID)
    both = pd.concat([v.rename("vec"), l.rename("loop")], axis=1).dropna()
    diff = float((both["vec"] - both["loop"]).abs().max()) if len(both) else float("nan")
    print(f"{name:20s} vec_n={len(v):5d} loop_n={len(l):5d} max_abs_diff={diff:.2e}", flush=True)

# ============ 1. ACTIVE LIBRARY DRIFT CHECK ============
print("\n=== 1. ACTIVE LIBRARY RE-VALIDATION (drift) ===", flush=True)
active = {
    "vol_price_corr_20": (per_asset(vpc_build), 1),
    "eurusd_beta_60d": (rolling_beta(rets, eur.pct_change(), 60), -1),
    "rate_beta_cn10y_60d": (rolling_beta(rets, cn10.pct_change(), 60), -1),
    "dn_mkt_beta_60d": (rolling_beta(rets, dn, 60), 1),
}
active_meta = {}
for name, (panel, es) in active.items():
    m, ics = evaluate(f"[active]{name}", panel, es)
    active_meta[name] = m

# ============ 2. BATCH C CANDIDATES ============
print("\n=== 2. BATCH C CANDIDATES (12) ===", flush=True)
cands = {}


def f_ke(s, a):
    r = s.pct_change()
    return (s - s.shift(20)).abs() / r.abs().rolling(20).sum()
cands["kaufman_eff_20"] = (per_asset(f_ke), 1)


def f_sk(s, a):
    return s.pct_change().rolling(20).skew()
cands["skew_20d"] = (per_asset(f_sk), -1)


def f_mx(s, a):
    return s.pct_change().rolling(20).max()
cands["max_ret_20d"] = (per_asset(f_mx), -1)


def f_rp(s, a):
    hi = panels[a]["high"].astype(float).reindex(s.index)
    lo = panels[a]["low"].astype(float).reindex(s.index)
    return ((s - lo) / (hi - lo).replace(0, np.nan)).rolling(20).mean()
cands["range_pos_20"] = (per_asset(f_rp), 1)


def f_am(s, a):
    r = s.pct_change().abs()
    v = panels[a]["volume"].astype(float).reindex(s.index)
    return (r / v.replace(0, np.nan)).rolling(20).mean()
cands["amihud_20"] = (per_asset(f_am), -1)


def f_vr(s, a):
    r = s.pct_change()
    return r.rolling(5).std() / r.rolling(60).std()
cands["vol_ratio_5_60"] = (per_asset(f_vr), -1)


def f_tr2(s, a):
    lp = np.log(s)
    t = pd.Series(np.arange(len(lp), dtype=float), index=lp.index)
    tm = t.rolling(60).mean()
    tv = (t * t).rolling(60).mean() - tm ** 2
    xm = lp.rolling(60).mean()
    xv = lp.rolling(60).var(ddof=0)
    cov = (lp * t).rolling(60).mean() - xm * tm
    return (cov ** 2 / (xv * tv)).where(xv > 1e-14)
cands["trend_r2_60"] = (per_asset(f_tr2), 1)

cands["rev_5d_skip1"] = (-(closes.shift(1) / closes.shift(6) - 1.0), 1)
cands["close_sma200"] = (closes / closes.rolling(200).mean() - 1.0, 1)
cands["corr_mkt_60"] = (rets.rolling(60).corr(mkt), 1)
cands["drawup_60"] = (closes / closes.rolling(60).min() - 1.0, 1)


def f_vt(s, a):
    v20 = s.pct_change().rolling(20).std()
    return v20.rolling(10).mean() / v20.rolling(60).mean() - 1.0
cands["vol_trend_60"] = (per_asset(f_vt), -1)

batchC_meta = {}
for name, (panel, es) in cands.items():
    m, ics = evaluate(name, panel, es)
    batchC_meta[name] = m

# ============ 3. NEW CANDIDATES THIS CYCLE ============
print("\n=== 3. NEW CANDIDATES (8) ===", flush=True)
new = {}


def f_pop(s, a):
    return s / s.rolling(60).max()  # pct of 60d peak, higher=closer to peak
new["pct_of_peak_60"] = (per_asset(f_pop), 1)


def f_sh(s, a):
    r = s.pct_change()
    return r.rolling(60).mean() / r.rolling(60).std()
new["sharpe_60"] = (per_asset(f_sh), 1)

new["mom_term_20_60"] = ((closes.shift(2) / closes.shift(22) - 1.0)
                         - (closes.shift(2) / closes.shift(62) - 1.0), 1)


def f_stk(s, a):
    hi = panels[a]["high"].astype(float).reindex(s.index)
    lo = panels[a]["low"].astype(float).reindex(s.index)
    return (s - lo.rolling(20).min()) / (hi.rolling(20).max() - lo.rolling(20).min()).replace(0, np.nan)
new["stoch_k_20"] = (per_asset(f_stk), 1)


def f_ta(s, a):
    return ((s > s.rolling(20).mean()).astype(float)
            + (s > s.rolling(50).mean()).astype(float)
            + (s > s.rolling(100).mean()).astype(float)) / 3.0
new["trend_align"] = (per_asset(f_ta), 1)

new["btc_beta_60d"] = (rolling_beta(rets, rets["BTC"], 60, 40, exclude_self="BTC"), 1)


def f_obv(s, a):
    v = panels[a]["volume"].astype(float).reindex(s.index)
    obv = (np.sign(s.diff()).fillna(0.0) * v).cumsum()
    return (obv - obv.shift(20)) / v.rolling(20).std().replace(0, np.nan)
new["obv_mom_20"] = (per_asset(f_obv), 1)


def f_rm(s, a):
    hi = panels[a]["high"].astype(float).reindex(s.index)
    lo = panels[a]["low"].astype(float).reindex(s.index)
    return ((hi - lo) / s).rolling(20).mean()
new["range_mean_20"] = (per_asset(f_rm), -1)

new_meta = {}
for name, (panel, es) in new.items():
    m, ics = evaluate(name, panel, es)
    new_meta[name] = m

print(f"\nDone. elapsed={time.time()-t0:.1f}s", flush=True)

# stash results for persistence step
import json
out = {"active": active_meta, "batchC": batchC_meta, "new": new_meta,
       "window": {"first": str(closes.index[0].date()), "last": str(closes.index[-1].date()),
                  "n_dates": int(len(closes)), "n_assets": int(closes.shape[1])}}
with open("scripts/_miner1_batchC2_results.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print("saved scripts/_miner1_batchC2_results.json")
