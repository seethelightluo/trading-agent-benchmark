"""miner_2 batch Q screen (2028-07-03) - vectorized rank-IC, no lookahead.

Validation window: 2020-01-01 .. 2028-06-30 (last completed trading day before
2028-07-03; data loaded via alphacrafter sim utils respects visible_through).

A) drift re-validation of 3 ACTIVE library factors
   active = vol_adj_mom_accel_20x60, dn_mkt_beta_60d, rate_beta_cn10y_60d
B) batch Q candidates (fresh families, low overlap with batches A-P):
   - Q_drawdown_60d      : close/rolling_max(close,60) - 1 (drawdown depth, reversal)
   - Q_dist_200d_ma      : close/rolling_mean(close,200) - 1 (trend position)
   - Q_breakout_20d      : close/rolling_max(close,20) - 1 (distance to recent high)
   - Q_recovery_rate_60d : fraction of positive days over 60d (win-rate persistence)
   - Q_skew20            : rolling skewness of 20d returns (short-horizon crash risk)
   - Q_kurt20            : rolling kurtosis of 20d returns (tail heaviness)
   - Q_dxy_beta_60d      : rolling beta of asset rets to DXY rets (USD regime)
   - Q_usdjpy_beta_60d   : rolling beta of asset rets to USDJPY rets (carry/risk)
   - Q_eurusd_beta_60d   : rolling beta of asset rets to EURUSD rets (prior effective)
   - Q_us10y_beta_chg_60_120 : beta60(asset,US10Y) - beta120(asset,US10Y) (rate-beta drift)
   - Q_range_20d         : mean((high-low)/close,20) (range-based vol)
   - Q_consec_down_20d   : max consecutive down-day streak over 20d
   - Q_gap_trend_20d     : mean(open/prev_close-1,20) (gap direction persistence)
   - Q_vix_beta_60d      : rolling beta of asset rets to VIX rets (non-conditional)

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

# macro panels (observation-only signals, used as conditioning/driver only)
dxy = panels["DXY"]["close"].astype(float)
usdjpy = panels["USDJPY"]["close"].astype(float)
eurusd = panels["EURUSD"]["close"].astype(float)
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


# ---------------- active library signals (reference for incrementality) ----------------
m20 = closes / closes.shift(20) - 1.0
m60 = closes / closes.shift(60) - 1.0
LIBRARY = {
    "vol_adj_mom_accel_20x60": (m20 - m60) / vol20.replace(0, np.nan),
    "dn_mkt_beta_60d": rolling_beta(rets, mkt.where(mkt < 0), 60, 40),
    "rate_beta_cn10y_60d": rolling_beta(rets, rets["CN10Y"], 60, 40),
}
print(f"library signals {time.time()-t0:.1f}s", flush=True)

# ---------------- batch Q candidate panels ----------------
dxy_ret = dxy.pct_change()
usdjpy_ret = usdjpy.pct_change()
eurusd_ret = eurusd.pct_change()
vix_ret = vix.pct_change()
high20 = highs.rolling(20).max()
low20 = lows.rolling(20).min()
pos = rets > 0

cands = {
    "Q_drawdown_60d": closes / closes.rolling(60).max() - 1.0,
    "Q_dist_200d_ma": closes / closes.rolling(200).mean() - 1.0,
    "Q_breakout_20d": closes / high20 - 1.0,
    "Q_recovery_rate_60d": pos.rolling(60).mean(),
    "Q_skew20": rets.rolling(20).skew(),
    "Q_kurt20": rets.rolling(20).kurt(),
    "Q_dxy_beta_60d": rolling_beta(rets, dxy_ret, 60, 40),
    "Q_usdjpy_beta_60d": rolling_beta(rets, usdjpy_ret, 60, 40),
    "Q_eurusd_beta_60d": rolling_beta(rets, eurusd_ret, 60, 40),
    "Q_us10y_beta_chg_60_120": rolling_beta(rets, rets["US10Y"], 60, 40) - rolling_beta(rets, rets["US10Y"], 120, 80),
    "Q_range_20d": ((highs - lows) / closes.replace(0, np.nan)).rolling(20).mean(),
    "Q_consec_down_20d": (rets < 0).rolling(20).apply(
        lambda x: int(pd.Series(x.astype(int)).groupby((x.astype(int) == 0).cumsum()).cumsum().max()) if len(x) else np.nan, raw=False),
    "Q_gap_trend_20d": (opens / closes.shift(1) - 1.0).rolling(20).mean(),
    "Q_vix_beta_60d": rolling_beta(rets, vix_ret, 60, 40),
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

# ---------------- batch Q screen ----------------
print("\n=== B) BATCH Q SCREEN (14 candidates, h=10) ===")
results = {}
for name, sig in cands.items():
    t1 = time.time()
    ics = rank_ic_series_fast(sig, fwd, MIN_VALID)
    m = summarize_fast(ics)
    cov = coverage_metrics(sig, min_valid=MIN_VALID)
    turn = turnover_rank(sig, 10)
    corr, key = 0.0, None
    best = 0.0
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
    print(f"{name:28s} {r['ic']:8.4f} {r['icir']:8.4f} {r['hit']:6.3f} {r['n']:5d} "
          f"{r['cov_asset_days']:7.3f} {r['cov_dates_ge8']:7.3f} {r['turnover_10d']:6.2f} "
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
