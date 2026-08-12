"""miner_3 batch Q screen (2029-03-12) - vectorized rank-IC, no lookahead.

A) drift re-validation of 3 ACTIVE library factors (full + recent 250/500/750 + decay)
   active = vol_adj_mom_accel_20x60, dn_mkt_beta_60d, rate_beta_cn10y_60d
B) batch Q candidates (fresh families, low overlap with batches A-P):
   - Q_dxy_beta_60d        : rolling 60d beta of asset ret to DXY ret (dollar sensitivity)
   - Q_dxy_beta_chg_20_60  : beta20(DXY) - beta60(DXY) (dollar-regime shift detector)
   - Q_usdjpy_beta_60d     : rolling 60d beta to USDJPY ret (carry/risk proxy)
   - Q_vix_beta_60d        : rolling 60d beta to VIX ret (unconditional risk-sensitivity)
   - Q_riskoff_cond_beta   : -beta60(asset,VIX) * riskoff_score (VIX z + DXY mom + USDJPY mom)
   - Q_intraday_pos_20d    : mean((close-open)/(high-low)) 20d (buying pressure)
   - Q_gap_drift_20d       : mean(open/prev_close-1) 20d (overnight gap persistence)
   - Q_range_20d           : mean((high-low)/close) 20d (realized range)
   - Q_drawdown_60d        : close/rolling_max(close,60)-1 (distance from 60d high)
   - Q_drawdown_120d       : close/rolling_max(close,120)-1 (distance from 120d high)
   - Q_trend_r2_60d        : rolling R^2 of 60d linear trend fit (trend quality)
   - Q_vol_curve_5_60      : vol5/vol60 - 1 (short-term vol acceleration)
   - Q_volume_ratio_5_60   : vol5/vol60 of volume - 1 (liquidity shift)
   - Q_mom_accel_5_20      : mom5/mom20 - 1 (price momentum acceleration)

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
                                 library_signals, TRADABLE)

t0 = time.time()
panels = load_panels(days=3000)
closes = close_panel(panels)
rets = closes.pct_change()
vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()
vol5 = rets.rolling(5).std()
opens = pd.DataFrame({a: panels[a]["open"].astype(float) for a in closes.columns}).reindex(closes.index)
highs = pd.DataFrame({a: panels[a]["high"].astype(float) for a in closes.columns}).reindex(closes.index)
lows = pd.DataFrame({a: panels[a]["low"].astype(float) for a in closes.columns}).reindex(closes.index)
vols = pd.DataFrame({a: panels[a]["volume"].astype(float) for a in closes.columns}).reindex(closes.index)
mkt = rets.mean(axis=1)

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


# ---------------- batch Q candidate panels ----------------
dxy_ret = dxy.pct_change()
usdjpy_ret = usdjpy.pct_change()
vix_ret = vix.pct_change()
riskoff = (vix - vix.rolling(60).mean()) / vix.rolling(60).std() \
    + dxy.pct_change(5) / dxy.pct_change(5).rolling(60).std() \
    + usdjpy.pct_change(5) / usdjpy.pct_change(5).rolling(60).std()
riskoff = riskoff.rolling(5).mean()  # smooth regime score (high = risk-off)

m5 = closes.pct_change(5)
m20 = closes.pct_change(20)
r2_60 = {}
for a in closes.columns:
    x = np.arange(60, dtype=float)
    def r2fit(y):
        if np.isnan(y).any():
            return np.nan
        b, a_ = np.polyfit(x, y, 1)
        yhat = a_ + b * x
        ss_res = np.nansum((y - yhat) ** 2)
        ss_tot = np.nansum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    r2_60[a] = closes[a].rolling(60).apply(r2fit, raw=True)
r2_60 = pd.DataFrame(r2_60, index=closes.index)

cands = {
    "Q_dxy_beta_60d": rolling_beta(rets, dxy_ret, 60, 40),
    "Q_dxy_beta_chg_20_60": rolling_beta(rets, dxy_ret, 20, 14) - rolling_beta(rets, dxy_ret, 60, 40),
    "Q_usdjpy_beta_60d": rolling_beta(rets, usdjpy_ret, 60, 40),
    "Q_vix_beta_60d": rolling_beta(rets, vix_ret, 60, 40),
    "Q_riskoff_cond_beta": -rolling_beta(rets, vix_ret, 60, 40).mul(riskoff, axis=0),
    "Q_intraday_pos_20d": ((closes - opens) / (highs - lows).replace(0, np.nan)).rolling(20).mean(),
    "Q_gap_drift_20d": (opens / closes.shift(1) - 1.0).rolling(20).mean(),
    "Q_range_20d": ((highs - lows) / closes).rolling(20).mean(),
    "Q_drawdown_60d": closes / closes.rolling(60).max() - 1.0,
    "Q_drawdown_120d": closes / closes.rolling(120).max() - 1.0,
    "Q_trend_r2_60d": r2_60,
    "Q_vol_curve_5_60": vol5 / vol60.replace(0, np.nan) - 1.0,
    "Q_volume_ratio_5_60": vols.rolling(5).mean() / vols.rolling(60).mean().replace(0, np.nan) - 1.0,
    "Q_mom_accel_5_20": m5 / m20.replace(0, np.nan) - 1.0,
}
print(f"candidate panels built {time.time()-t0:.1f}s", flush=True)

# ---------------- active library signals ----------------
LIBRARY = library_signals(panels, closes=closes, rets=rets, vix=vix)
# recompute the 3 active factors precisely
lib = {}
lib["vol_adj_mom_accel_20x60"] = (m20 / vol20.replace(0, np.nan)) - (m5 / vol5.replace(0, np.nan))
mkt_beta = rolling_beta(rets, mkt, 60, 40)
lib["dn_mkt_beta_60d"] = mkt_beta.where(mkt < 0)
cn10y_ret = rets["CN10Y"]
lib["rate_beta_cn10y_60d"] = rolling_beta(rets, cn10y_ret, 60, 40)
ACTIVE = {k: lib[k] for k in ["vol_adj_mom_accel_20x60", "dn_mkt_beta_60d", "rate_beta_cn10y_60d"]}
LIBRARY_FULL = dict(LIBRARY)
LIBRARY_FULL.update(ACTIVE)

fwd = forward_returns(closes, H_ADM)

# ---------------- A) active library drift ----------------
print("\n=== A) ACTIVE LIBRARY DRIFT (h=10) ===")
for name, sig in ACTIVE.items():
    ics_full = rank_ic_series_fast(sig, fwd, MIN_VALID)
    m_full = summarize_fast(ics_full)
    parts = [f"{name}: full_ic={m_full['ic']:.4f} icir={m_full['icir']:.4f} hit={m_full['ic_hit_ratio']:.3f} n={m_full['n_ic_dates']}"]
    for wname, w in [("r250", 250), ("r500", 500), ("r750", 750)]:
        if len(ics_full) > w:
            sub = ics_full.iloc[-w:]
            ms = summarize_fast(sub)
            parts.append(f"| {wname}: ic={ms['ic']:.4f} icir={ms['icir']:.4f}")
        else:
            parts.append(f"| {wname}: n/a")
    print(" ".join(parts), flush=True)

# ---------------- B) batch Q screen ----------------
print("\n=== B) BATCH Q SCREEN (14 candidates, h=10) ===")
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

print("\n=== RECENT WINDOW + DECAY (ic by horizon) for passers & near-passers ===")
watch = [n for n, r in results.items() if (abs(r["ic"]) >= GATE_IC or abs(r["icir"]) >= GATE_ICIR)]
for name in watch:
    r = results[name]
    dec = r["decay"]
    print(f"{name}: " + " ".join(f"h{h}={dec[str(h)]:.4f}" for h in [1, 2, 3, 5, 10, 20])
          + f" | r250 ic={r.get('ic_r250', np.nan):.4f} icir={r.get('icir_r250', np.nan):.4f}"
          + f" | r500 ic={r.get('ic_r500', np.nan):.4f} icir={r.get('icir_r500', np.nan):.4f}", flush=True)

print("\nPASSERS (ic+icir+robust+incremental):", passers)
print(f"elapsed {time.time()-t0:.1f}s")
