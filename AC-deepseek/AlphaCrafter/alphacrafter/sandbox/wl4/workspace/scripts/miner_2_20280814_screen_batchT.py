"""miner_2 batch T screen (2028-08-14) - vectorized rank-IC, no lookahead.

Validation window: 2020-01-01 .. 2028-08-11 (last completed trading day per
persistent/date.json visible_through; data loaded via alphacrafter sim utils).

A) drift re-validation of 3 ACTIVE library factors
   active = vol_adj_mom_accel_20x60, dn_mkt_beta_60d, rate_beta_cn10y_60d
B) batch T candidates (fresh families, low overlap with batches A-S):
   - T_ema_ratio_10_40     : EMA(close,10)/EMA(close,40) - 1 (EMA trend cross)
   - T_trend_quality_20d   : R^2 of rolling 20d OLS on log close (trend quality)
   - T_reversal_5d_vol     : -(close/close.shift(5)-1) / vol20 (vol-scaled short-term reversal)
   - T_vol_zs_120d         : vol20 / rolling_mean(vol20,120) - 1 (vol regime vs 6m mean)
   - T_dn_vol_share_40d    : std(neg rets)/std(all rets) over 40d (downside risk share)
   - T_corr_dispersion_20d : mean pairwise return correlation vs other 14 assets (systemic linkage)
   - T_xau_up_beta_60d     : rolling beta of asset rets to XAU rets on XAU-up days (gold up-capture)
   - T_dxy_beta_chg_60_120 : beta60(asset,DXY) - beta120(asset,DXY) (USD sensitivity drift)
   - T_hi_lo_pos_20d       : mean((close-low)/(high-low),20) (intraday strength)
   - T_avg_gain_loss_20d   : mean positive ret / |mean negative ret| over 20d (RSI-like ratio)
   - T_beta_mkt_120d       : plain rolling market beta 120d (unconditional)
   - T_yc_spread_beta_60d  : beta of asset rets to (US10Y-CN10Y) spread changes
   - T_vix_regime_mom_20d  : mom20 * sign(VIX < VIX.rolling(60).median() ? +1 : -1)
                             (regime-conditional momentum: trend in calm, reversal in stress)
   - T_skew_zs_120d        : (skew20 - mean(skew20,120)) / std(skew20,120) (skew regime z-score)

Gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10 (15-asset universe, min_valid=8).
Incremental: max_abs_library_correlation < 0.5 vs ACTIVE library signals.
Robustness: n_ic_dates >= 30 required.
"""
import sys, time
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from factor_research_lib import load_panels, close_panel, forward_returns, coverage_metrics, turnover_rank

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


def rolling_r2_logprice(close, win=20):
    """Rolling R^2 of linear fit of log close on time trend."""
    lc = np.log(close)
    out = {}
    for a in close.columns:
        s = lc[a]
        x = np.arange(win)
        xm = x.mean()
        xd = x - xm
        def _r2(w):
            y = w.values
            ym = y.mean()
            yd = y - ym
            b = (xd * yd).sum() / (xd * xd).sum()
            yhat = ym + b * xd
            ss_res = ((y - yhat) ** 2).sum()
            ss_tot = ((y - ym) ** 2).sum()
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        out[a] = s.rolling(win).apply(_r2, raw=False)
    return pd.DataFrame(out, index=close.index)


# ---------------- active library signals (reference for incrementality) ----------------
m20 = closes / closes.shift(20) - 1.0
m60 = closes / closes.shift(60) - 1.0
LIBRARY = {
    "vol_adj_mom_accel_20x60": (m20 - m60) / vol20.replace(0, np.nan),
    "dn_mkt_beta_60d": rolling_beta(rets, mkt.where(mkt < 0), 60, 40),
    "rate_beta_cn10y_60d": rolling_beta(rets, rets["CN10Y"], 60, 40),
}
print(f"library signals {time.time()-t0:.1f}s", flush=True)

# ---------------- batch T candidate panels ----------------
dxy_ret = dxy.pct_change()
vix_ret = vix.pct_change()
xau_ret = rets["XAU"]
yc_spread = rets["US10Y"] - rets["CN10Y"]          # yield-spread daily change proxy
ema10 = closes.ewm(span=10, adjust=False).mean()
ema40 = closes.ewm(span=40, adjust=False).mean()
neg_rets = rets.where(rets < 0, 0.0)
pos_rets = rets.where(rets > 0, 0.0)

# pairwise correlation dispersion (rolling 20d mean pairwise corr per asset)
def rolling_mean_pairwise_corr(asset_ret, win=20):
    out = {}
    for a in asset_ret.columns:
        others = [c for c in asset_ret.columns if c != a]
        corrs = []
        for b in others:
            z = pd.concat([asset_ret[a].rename("a"), asset_ret[b].rename("b")], axis=1).dropna()
            corrs.append(z["a"].rolling(win).corr(z["b"]))
        out[a] = pd.concat(corrs, axis=1).mean(axis=1)
    return pd.DataFrame(out, index=asset_ret.index)

skew20 = rets.rolling(20).skew()

cands = {
    "T_ema_ratio_10_40": ema10 / ema40 - 1.0,
    "T_trend_quality_20d": rolling_r2_logprice(closes, 20),
    "T_reversal_5d_vol": -(closes / closes.shift(5) - 1.0) / vol20.replace(0, np.nan),
    "T_vol_zs_120d": vol20 / vol20.rolling(120).mean() - 1.0,
    "T_dn_vol_share_40d": neg_rets.rolling(40).std() / rets.rolling(40).std(),
    "T_corr_dispersion_20d": rolling_mean_pairwise_corr(rets, 20),
    "T_xau_up_beta_60d": rolling_beta(rets, xau_ret.where(xau_ret > 0), 60, 40),
    "T_dxy_beta_chg_60_120": rolling_beta(rets, dxy_ret, 60, 40) - rolling_beta(rets, dxy_ret, 120, 80),
    "T_hi_lo_pos_20d": ((closes - lows) / (highs - lows).replace(0, np.nan)).rolling(20).mean(),
    "T_avg_gain_loss_20d": pos_rets.rolling(20).mean() / neg_rets.rolling(20).mean().abs().replace(0, np.nan),
    "T_beta_mkt_120d": rolling_beta(rets, mkt, 120, 80),
    "T_yc_spread_beta_60d": rolling_beta(rets, yc_spread, 60, 40),
    "T_vix_regime_mom_20d": m20 * np.sign(vix.rolling(60).median() - vix).reindex(closes.index).fillna(1.0),
    "T_skew_zs_120d": (skew20 - skew20.rolling(120).mean()) / skew20.rolling(120).std().replace(0, np.nan),
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

# ---------------- batch T screen ----------------
print("\n=== B) BATCH T SCREEN (14 candidates, h=10) ===")
results = {}
for name, sig in cands.items():
    t1 = time.time()
    ics = rank_ic_series_fast(sig, fwd, MIN_VALID)
    m = summarize_fast(ics)
    cov = coverage_metrics(sig, min_valid=MIN_VALID)
    turn = turnover_rank(sig, 10)
    best, key = 0.0, None
    for lname, lsig in LIBRARY.items():
        both = pd.concat([sig.stack().rename("cand"), lsig.stack().rename("lib")], axis=1).dropna()
        if len(both) < 30:
            continue
        r = float(both["cand"].corr(both["lib"]))
        if abs(r) > best:
            best, key = abs(r), lname
    corr = round(best, 4)
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
hdr = f"{'factor':28s} {'ic':>8s} {'icir':>8s} {'hit':>6s} {'n':>5s} {'cov_ad':>7s} {'cov_d8':>7s} {'turn':>6s} {'lib_corr':>8s} {'pass_ic':>7s} {'pass_all':>8s}"
print(hdr)
passers = []
for name, r in results.items():
    pass_ic = (abs(r["ic"]) >= GATE_IC) and (abs(r["icir"]) >= GATE_ICIR)
    robust = r["n"] >= 30
    incr = r["max_lib_corr"] < 0.5
    pass_all = pass_ic and robust and incr
    if pass_all:
        passers.append(name)
    turn_s = f"{r['turnover_10d']:6.2f}" if r['turnover_10d'] is not None else "   NA "
    print(f"{name:28s} {r['ic']:8.4f} {r['icir']:8.4f} {r['hit']:6.3f} {r['n']:5d} "
          f"{r['cov_asset_days']:7.3f} {r['cov_dates_ge8']:7.3f} {turn_s} "
          f"{r['max_lib_corr']:8.3f} {str(pass_ic):>7s} {str(pass_all):>8s}", flush=True)

print("\n=== DECAY (ic by horizon) for passers ===")
for name in passers:
    r = results[name]
    dec = r["decay"]
    print(f"{name}: " + " ".join(f"h{h}={dec[str(h)]:.4f}" for h in [1, 2, 3, 5, 10, 20]), flush=True)

print("\n=== RECENT WINDOW IC for passers ===")
for name in passers:
    r = results[name]
    print(f"{name}: r250 ic={r.get('ic_r250', np.nan):.4f} icir={r.get('icir_r250', np.nan):.4f} | "
          f"r500 ic={r.get('ic_r500', np.nan):.4f} icir={r.get('icir_r500', np.nan):.4f} | "
          f"r750 ic={r.get('ic_r750', np.nan):.4f} icir={r.get('icir_r750', np.nan):.4f}", flush=True)

print("\nPASSERS (ic+icir+robust+incremental):", passers)
print(f"elapsed {time.time()-t0:.1f}s")
