"""miner_3 batch R screen (2029-10-08) - vectorized rank-IC, no lookahead.

A) drift re-validation of 3 ACTIVE library factors (full + recent 250/500/750 + decay)
   active = vol_adj_mom_accel_20x60, dn_mkt_beta_60d, rate_beta_cn10y_60d
B) batch R candidates (fresh families, low overlap with batches A-Q):
   - R_corr_mkt_60d       : rolling 60d correlation of asset ret to equal-weight mkt ret
   - R_corr_mkt_chg_20_60 : corr20(mkt) - corr60(mkt) (regime-coupling shift)
   - R_rel_strength_20    : (1+mom20_asset)/(1+mom20_mkt) - 1 (relative strength)
   - R_downside_vol_share_60: std(neg days)/std(all days) over 60d
   - R_efficiency_60      : |net move 60d| / sum(|daily ret|) 60d (trend efficiency)
   - R_winrate_60         : fraction of positive daily returns over 60d
   - R_skew_60            : rolling skewness of daily returns, 60d
   - R_kurt_60            : rolling excess kurtosis of daily returns, 60d
   - R_autocorr_5_60      : rolling 5-lag autocorrelation of returns (60d window)
   - R_tail_ratio_60      : mean(top5 ret)/abs(mean(bottom5 ret)) over 60d
   - R_crypto_beta_60     : rolling 60d beta to BTC returns
   - R_xau_beta_60        : rolling 60d beta to XAU returns (safe-haven sensitivity)
   - R_curve_beta_60      : rolling 60d beta to (US10Y-CN10Y) spread change
   - R_vol_updown_diff_60 : vol(up days) - vol(down days) over 60d (asymmetry)
   - R_dd_recovery_60     : close/rolling_min(close,60) - 1 (distance from 60d low)

Gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10 (15-asset universe, min_valid=8).
Incremental: max_abs_library_correlation < 0.5 vs library signals.
Robustness: n_ic_dates >= 30 required.
"""
import sys, time, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 coverage_metrics, turnover_rank, max_library_corr,
                                 library_signals, TRADABLE)

t0 = time.time()
panels = load_panels(days=3000)
closes = close_panel(panels)
rets = closes.pct_change()
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()
mkt = rets.mean(axis=1)
mkt_mom20 = closes.mean(axis=1).pct_change(20)

dxy = panels["DXY"]["close"].astype(float)
usdjpy = panels["USDJPY"]["close"].astype(float)
vix = panels["VIX"]["close"].astype(float)
print(f"panels loaded {time.time()-t0:.1f}s | closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()}", flush=True)
print("last completed trading day:", closes.index.max().date(), flush=True)

H_ADM = 10
MIN_VALID = 8
GATE_IC, GATE_ICIR = 0.0070, 0.0840


def rank_ic_series_fast(factor_panel: pd.DataFrame, fwd: pd.DataFrame, min_valid: int = 8) -> pd.Series:
    rf = factor_panel.rank(axis=1, method="average")
    rr = fwd.rank(axis=1, method="average")
    valid = rf.notna() & rr.notna()
    nv = valid.sum(axis=1)
    rf2 = rf.where(valid)
    rr2 = rr.where(valid)
    mu_f = rf2.sum(axis=1) / nv.replace(0, np.nan)
    mu_r = rr2.sum(axis=1) / nv.replace(0, np.nan)
    cf = rf2.sub(mu_f, axis=0).fillna(0.0)
    cr = rr2.sub(mu_r, axis=0).fillna(0.0)
    ssf = (cf ** 2).sum(axis=1).astype(float)
    ssr = (cr ** 2).sum(axis=1).astype(float)
    cov = (cf * cr).sum(axis=1).astype(float)
    den = np.sqrt((ssf * ssr).replace([0.0, np.inf, -np.inf], np.nan).to_numpy(dtype=float))
    icv = (cov.to_numpy(dtype=float) / den)
    ic = pd.Series(icv, index=cov.index)
    ok = (nv >= min_valid) & (ssf > 1e-14) & (ssr > 1e-14) & ic.notna()
    return ic[ok].rename("ic")


def summarize_fast(ic_series: pd.Series):
    ic = float(ic_series.mean())
    sd = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else 0.0
    icir = ic / sd if sd > 0 else 0.0
    return {"ic": ic, "icir": icir, "ic_hit_ratio": float((ic_series > 0).mean()),
            "n_ic_dates": int(len(ic_series)), "ic_std": sd}


def decay_profile(factor_panel, horizons=(1, 2, 3, 5, 10, 20)):
    out = {}
    for h in horizons:
        fwd = forward_returns(closes, h)
        ics = rank_ic_series_fast(factor_panel, fwd, MIN_VALID)
        out[str(h)] = round(float(ics.mean()), 4) if len(ics) else np.nan
    return out


def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40):
    beta = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        cov = z["a"].rolling(win).cov(z["m"])
        var = z["m"].rolling(win).var()
        beta[a] = (cov / var).where(z["m"].rolling(win).count() >= min_obs)
    return pd.DataFrame(beta, index=asset_ret.index)


def rolling_corr(asset_ret, driver_ret, win=60, min_obs=40):
    out = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        c = z["a"].rolling(win).corr(z["m"])
        out[a] = c.where(z["m"].rolling(win).count() >= min_obs)
    return pd.DataFrame(out, index=asset_ret.index)


# ---------------- batch R candidate panels ----------------
mkt_corr60 = rolling_corr(rets, mkt, 60, 40)
mkt_corr20 = rolling_corr(rets, mkt, 20, 14)
neg = rets.where(rets < 0)
pos = rets.where(rets > 0)
vol_neg60 = neg.rolling(60).std()
vol_pos60 = pos.rolling(60).std()
sum_abs60 = rets.abs().rolling(60).sum()
skew60 = rets.rolling(60).skew()
kurt60 = rets.rolling(60).kurt()
autocorr5 = rets.rolling(60).apply(lambda x: pd.Series(x).autocorr(5), raw=False)
top5 = rets.rolling(60).apply(lambda x: np.mean(np.sort(x)[-5:]), raw=True)
bot5 = rets.rolling(60).apply(lambda x: np.mean(np.sort(x)[:5]), raw=True)
btc_ret = rets["BTC"]
xau_ret = rets["XAU"]
curve_spread = closes["US10Y"] - closes["CN10Y"]

cands = {
    "R_corr_mkt_60d": mkt_corr60,
    "R_corr_mkt_chg_20_60": mkt_corr20 - mkt_corr60,
    "R_rel_strength_20": (1 + closes.pct_change(20)) / (1 + mkt_mom20) - 1.0,
    "R_downside_vol_share_60": vol_neg60 / vol60.replace(0, np.nan),
    "R_efficiency_60": (closes / closes.shift(60) - 1.0).abs() / sum_abs60.replace(0, np.nan),
    "R_winrate_60": (rets > 0).rolling(60).mean(),
    "R_skew_60": skew60,
    "R_kurt_60": kurt60,
    "R_autocorr_5_60": autocorr5,
    "R_tail_ratio_60": top5 / bot5.replace(0, np.nan).abs(),
    "R_crypto_beta_60": rolling_beta(rets, btc_ret, 60, 40),
    "R_xau_beta_60": rolling_beta(rets, xau_ret, 60, 40),
    "R_curve_beta_60": rolling_beta(rets, curve_spread.pct_change(), 60, 40),
    "R_vol_updown_diff_60": vol_pos60 - vol_neg60,
    "R_dd_recovery_60": closes / closes.rolling(60).min() - 1.0,
}
print(f"candidate panels built {time.time()-t0:.1f}s", flush=True)

# ---------------- library signals ----------------
LIBRARY = library_signals(panels, closes=closes, rets=rets, vix=vix)
m5 = closes.pct_change(5)
m20 = closes.pct_change(20)
lib = {}
lib["vol_adj_mom_accel_20x60"] = (m20 / vol20.replace(0, np.nan)) - (m5 / rets.rolling(5).std().replace(0, np.nan))
mkt_beta = rolling_beta(rets, mkt, 60, 40)
lib["dn_mkt_beta_60d"] = mkt_beta.where(mkt < 0)
lib["rate_beta_cn10y_60d"] = rolling_beta(rets, rets["CN10Y"], 60, 40)
ACTIVE = {k: lib[k] for k in ["vol_adj_mom_accel_20x60", "dn_mkt_beta_60d", "rate_beta_cn10y_60d"]}
LIBRARY_FULL = dict(LIBRARY)
LIBRARY_FULL.update(ACTIVE)

fwd = forward_returns(closes, H_ADM)

# ---------------- A) active library drift ----------------
print("\n=== A) ACTIVE LIBRARY DRIFT (h=10) ===")
drift = {}
for name, sig in ACTIVE.items():
    ics_full = rank_ic_series_fast(sig, fwd, MIN_VALID)
    m_full = summarize_fast(ics_full)
    parts = [f"{name}: full_ic={m_full['ic']:.4f} icir={m_full['icir']:.4f} hit={m_full['ic_hit_ratio']:.3f} n={m_full['n_ic_dates']}"]
    row = {"full": m_full}
    for wname, w in [("r250", 250), ("r500", 500), ("r750", 750)]:
        if len(ics_full) > w:
            sub = ics_full.iloc[-w:]
            ms = summarize_fast(sub)
            row[wname] = ms
            parts.append(f"| {wname}: ic={ms['ic']:.4f} icir={ms['icir']:.4f}")
        else:
            parts.append(f"| {wname}: n/a")
    drift[name] = row
    print(" ".join(parts), flush=True)

# ---------------- B) batch R screen ----------------
print("\n=== B) BATCH R SCREEN (15 candidates, h=10) ===")
results = {}
for name, sig in cands.items():
    t1 = time.time()
    ics = rank_ic_series_fast(sig, fwd, MIN_VALID)
    m = summarize_fast(ics)
    cov = coverage_metrics(sig, min_valid=MIN_VALID)
    turn = turnover_rank(sig, 10)
    corr, key = max_library_corr(sig, LIBRARY_FULL)
    dec = decay_profile(sig)
    row = {"ic": m["ic"], "icir": m["icir"], "hit": m["ic_hit_ratio"], "n": m["n_ic_dates"],
           "cov_asset_days": cov["coverage_asset_days"], "cov_dates_ge8": cov["coverage_dates_ge8"],
           "turnover_10d": turn, "max_lib_corr": corr, "max_corr_factor": key, "decay": dec}
    for wname, w in [("r250", 250), ("r500", 500), ("r750", 750)]:
        if len(ics) > w:
            sub = ics.iloc[-w:]
            ms = summarize_fast(sub)
            row[f"ic_{wname}"] = ms["ic"]
            row[f"icir_{wname}"] = ms["icir"]
        else:
            row[f"ic_{wname}"] = np.nan
            row[f"icir_{wname}"] = np.nan
    results[name] = row
    print(f"done {name} {time.time()-t1:.1f}s", flush=True)

print("\n=== FULL SCREEN (h=10, min_valid=8) ===")
hdr = f"{'factor':26s} {'ic':>8s} {'icir':>8s} {'hit':>6s} {'n':>5s} {'cov_ad':>7s} {'cov_d8':>7s} {'turn':>6s} {'lib_corr':>8s} {'pass_ic':>7s} {'pass_all':>8s}"
print(hdr)
passers = []
for name, r in results.items():
    pass_ic = (abs(r["ic"]) >= GATE_IC) and (abs(r["icir"]) >= GATE_ICIR)
    robust = r["n"] >= 30
    incr = r["max_lib_corr"] < 0.5
    pass_all = pass_ic and robust and incr
    if pass_all:
        passers.append(name)
    print(f"{name:26s} {r['ic']:8.4f} {r['icir']:8.4f} {r['hit']:6.3f} {r['n']:5d} "
          f"{r['cov_asset_days']:7.3f} {r['cov_dates_ge8']:7.3f} {r['turnover_10d']:6.2f} "
          f"{r['max_lib_corr']:8.3f} {str(pass_ic):>7s} {str(pass_all):>8s}", flush=True)

print("\n=== DECAY (ic by horizon) for passers & near-passers ===")
watch = [n for n, r in results.items() if (abs(r["ic"]) >= GATE_IC or abs(r["icir"]) >= GATE_ICIR)]
for name in watch:
    r = results[name]
    dec = r["decay"]
    print(f"{name}: " + " ".join(f"h{h}={dec[str(h)]:.4f}" for h in [1, 2, 3, 5, 10, 20])
          + f" | r250 ic={r.get('ic_r250', np.nan):.4f} icir={r.get('icir_r250', np.nan):.4f}"
          + f" | r500 ic={r.get('ic_r500', np.nan):.4f} icir={r.get('icir_r500', np.nan):.4f}", flush=True)

# regime snapshot
print("\n=== regime snapshot ===")
mkt_ret = rets.mean(axis=1)
for w in [20, 60]:
    r = (1 + mkt_ret).rolling(w).apply(np.prod, raw=True) - 1
    v = mkt_ret.rolling(w).std() * np.sqrt(252)
    print(f"mkt(live) {w:3d}d cum: {r.iloc[-1]*100:+.2f}%  vol_ann: {v.iloc[-1]*100:.1f}%")
print("VIX last:", round(float(vix.iloc[-1]), 2), " 60d ago:", round(float(vix.iloc[-61]), 2))
print("DXY last:", round(float(dxy.iloc[-1]), 2), " 60d ago:", round(float(dxy.iloc[-61]), 2))
print("USDJPY last:", round(float(usdjpy.iloc[-1]), 2), " 60d ago:", round(float(usdjpy.iloc[-61]), 2))

out = {"drift": drift, "results": results, "passers": passers,
       "last_date": str(closes.index.max().date())}
json.dump(out, open("scripts/_miner3_20291008_batchR_results.json", "w"), indent=1, default=str)
print("\nPASSERS (ic+icir+robust+incremental):", passers)
print(f"saved scripts/_miner3_20291008_batchR_results.json | elapsed {time.time()-t0:.1f}s")
