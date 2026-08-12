"""miner_3 batch P screen (2028-06-05) - vectorized rank-IC, no lookahead.

A) drift re-validation of 3 ACTIVE library factors (full + recent 250/500/750 + decay)
   active = vol_adj_mom_accel_20x60, dn_mkt_beta_60d, rate_beta_cn10y_60d
B) batch P candidates (fresh families, low overlap with batches A-O):
   - P_stoch_pos_20d      : (close - min20)/(max20 - min20) range position
   - P_skew_roll_60d      : rolling skewness of 60d daily returns (crash-risk proxy)
   - P_tail_ratio_60d     : p95(|ret|)/p50(|ret|) over 60d (tail heaviness)
   - P_updown_semivol_60d : std(up rets)/std(dn rets) over 60d (asymmetric risk)
   - P_var_ratio_60_20    : var60/(3*var20) - 1 (trend vs mean-reversion signature)
   - P_autocorr_20d       : rolling 1-lag return autocorrelation (persistence)
   - P_max_ret_20d        : max daily return over 20d (lottery/MAX effect)
   - P_btc_beta_60d       : rolling beta of asset rets to BTC rets (crypto exposure)
   - P_xau_beta_60d       : rolling beta of asset rets to XAU rets (real-asset exposure)
   - P_volbeta_60d        : rolling beta of asset vol20 to market vol20 (vol sensitivity)
   - P_mktcorr_chg_20_60  : corr20(asset,mkt) - corr60(asset,mkt) (linkage shift)
   - P_asm_corr_60d       : corr(asset,mkt|mkt>0) - corr(asset,mkt|mkt<0) (asym corr)
   - P_mom_sharpe_60d     : mom60 / vol60 (risk-adjusted trend)
   - P_gap_freq_20d       : fraction of days with |open/prev_close-1|>1% over 20d

Gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10 (15-asset universe, min_valid=8).
Incremental: max_abs_library_correlation < 0.5 vs active library signals.
Robustness: n_ic_dates >= 30 required.
"""
import sys, time, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 coverage_metrics, turnover_rank, max_library_corr,
                                 TRADABLE)

t0 = time.time()
panels = load_panels(days=3000)
closes = close_panel(panels)
rets = closes.pct_change()
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()
opens = pd.DataFrame({a: panels[a]["open"].astype(float) for a in closes.columns}).reindex(closes.index)
highs = pd.DataFrame({a: panels[a]["high"].astype(float) for a in closes.columns}).reindex(closes.index)
lows = pd.DataFrame({a: panels[a]["low"].astype(float) for a in closes.columns}).reindex(closes.index)
mkt = rets.mean(axis=1)
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


def rolling_corr_fast(a, b, win=20, min_obs=15):
    n = a.rolling(win).count()
    cov = (a * b).rolling(win).mean() - a.rolling(win).mean() * b.rolling(win).mean()
    den = a.rolling(win).std() * b.rolling(win).std()
    return (cov / den.replace(0, np.nan)).where(n >= min_obs)


# ---------------- active library signals (reference for incrementality) ----------------
m20 = closes / closes.shift(20) - 1.0
m60 = closes / closes.shift(60) - 1.0
LIBRARY = {
    "vol_adj_mom_accel_20x60": (m20 - m60) / vol20.replace(0, np.nan),
    "dn_mkt_beta_60d": rolling_beta(rets, mkt.where(mkt < 0), 60, 40),
    "rate_beta_cn10y_60d": rolling_beta(rets, rets["CN10Y"], 60, 40),
}
print(f"library signals {time.time()-t0:.1f}s", flush=True)

# ---------------- batch P candidate panels ----------------
roll_skew = rets.rolling(60).skew()
abs_ret = rets.abs()
tail_ratio = abs_ret.rolling(60).quantile(0.95) / abs_ret.rolling(60).quantile(0.50).replace(0, np.nan)
up = rets.where(rets > 0)
dn = rets.where(rets < 0)
updown_semivol = up.rolling(60).std() / dn.rolling(60).std().abs().replace(0, np.nan)
var20 = rets.rolling(20).var()
var60 = rets.rolling(60).var()
max_ret20 = rets.rolling(20).max()
mkt_vol20 = mkt.rolling(20).std()
mkt_vol60 = mkt.rolling(60).std()
gap = (opens / closes.shift(1) - 1.0).abs()

cands = {
    "P_stoch_pos_20d": (closes - lows.rolling(20).min()) / (highs.rolling(20).max() - lows.rolling(20).min()).replace(0, np.nan),
    "P_skew_roll_60d": roll_skew,
    "P_tail_ratio_60d": tail_ratio,
    "P_updown_semivol_60d": updown_semivol,
    "P_var_ratio_60_20": var60 / (3.0 * var20).replace(0, np.nan) - 1.0,
    "P_autocorr_20d": pd.DataFrame({a: rets[a].rolling(20).apply(lambda x: pd.Series(x).autocorr(1) if len(x) > 5 else np.nan, raw=False) for a in rets.columns}, index=rets.index),
    "P_max_ret_20d": max_ret20,
    "P_btc_beta_60d": rolling_beta(rets, rets["BTC"], 60, 40),
    "P_xau_beta_60d": rolling_beta(rets, rets["XAU"], 60, 40),
    "P_volbeta_60d": rolling_beta(vol20, mkt_vol20, 60, 40),
    "P_mktcorr_chg_20_60": rolling_corr_fast(rets, mkt, 20, 15) - rolling_corr_fast(rets, mkt, 60, 40),
    "P_asm_corr_60d": rolling_corr_fast(rets.where(mkt > 0), mkt, 60, 30) - rolling_corr_fast(rets.where(mkt < 0), mkt, 60, 30),
    "P_mom_sharpe_60d": m60 / vol60.replace(0, np.nan),
    "P_gap_freq_20d": (gap > 0.01).rolling(20).mean(),
}
print(f"candidate panels built {time.time()-t0:.1f}s", flush=True)

fwd = forward_returns(closes, H_ADM)

# ---------------- active library drift ----------------
print("\n=== A) ACTIVE LIBRARY DRIFT (h=10) ===")
active_rows = {}
for name, sig in LIBRARY.items():
    ics_full = rank_ic_series_fast(sig, fwd, MIN_VALID)
    m_full = summarize_fast(ics_full)
    row = {"ic": m_full["ic"], "icir": m_full["icir"], "hit": m_full["ic_hit_ratio"], "n": m_full["n_ic_dates"]}
    for wname, w in [("r250", 250), ("r500", 500), ("r750", 750)]:
        if len(ics_full) > w:
            sub = ics_full.iloc[-w:]
            ms = summarize_fast(sub)
            row[f"ic_{wname}"] = ms["ic"]
            row[f"icir_{wname}"] = ms["icir"]
        else:
            row[f"ic_{wname}"] = np.nan
            row[f"icir_{wname}"] = np.nan
    active_rows[name] = row
    print(f"{name}: full_ic={row['ic']:.4f} icir={row['icir']:.4f} hit={row['hit']:.3f} n={row['n']} "
          f"| r250: ic={row.get('ic_r250', np.nan):.4f} icir={row.get('icir_r250', np.nan):.4f} "
          f"| r500: ic={row.get('ic_r500', np.nan):.4f} icir={row.get('icir_r500', np.nan):.4f} "
          f"| r750: ic={row.get('ic_r750', np.nan):.4f} icir={row.get('icir_r750', np.nan):.4f}", flush=True)

# ---------------- batch P screen ----------------
print("\n=== B) BATCH P SCREEN (14 candidates, h=10) ===")
results = {}
for name, sig in cands.items():
    t1 = time.time()
    ics = rank_ic_series_fast(sig, fwd, MIN_VALID)
    m = summarize_fast(ics)
    cov = coverage_metrics(sig, min_valid=MIN_VALID)
    turn = turnover_rank(sig, 10)
    corr, key = max_library_corr(sig, LIBRARY)
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
hdr = f"{'factor':24s} {'ic':>8s} {'icir':>8s} {'hit':>6s} {'n':>5s} {'cov_ad':>7s} {'cov_d8':>7s} {'turn':>6s} {'lib_corr':>8s} {'pass_ic':>7s} {'pass_all':>8s}"
print(hdr)
passers = []
for name, r in results.items():
    pass_ic = (abs(r["ic"]) >= GATE_IC) and (abs(r["icir"]) >= GATE_ICIR)
    robust = r["n"] >= 30
    incr = r["max_lib_corr"] < 0.5
    pass_all = pass_ic and robust and incr
    if pass_all:
        passers.append(name)
    print(f"{name:24s} {r['ic']:8.4f} {r['icir']:8.4f} {r['hit']:6.3f} {r['n']:5d} "
          f"{r['cov_asset_days']:7.3f} {r['cov_dates_ge8']:7.3f} {r['turnover_10d']:6.2f} "
          f"{r['max_lib_corr']:8.3f} {str(pass_ic):>7s} {str(pass_all):>8s}", flush=True)

print("\n=== DECAY (ic by horizon) for passers ===")
for name in passers:
    r = results[name]
    dec = r["decay"]
    print(f"{name}: " + " ".join(f"h{h}={dec[str(h)]:.4f}" for h in [1, 2, 3, 5, 10, 20]), flush=True)

print("\nPASSERS (ic+icir+robust+incremental):", passers)
print(f"elapsed {time.time()-t0:.1f}s")
