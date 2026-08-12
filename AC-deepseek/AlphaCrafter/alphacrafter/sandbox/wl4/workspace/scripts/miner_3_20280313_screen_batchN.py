"""miner_3 batch N screen (2028-03-13) - vectorized rank-IC, no lookahead.

A) drift re-validation of 3 ACTIVE library factors (full + recent 250/500/750)
   active = vol_price_corr_20, dn_mkt_beta_60d, rate_beta_cn10y_60d
   (eurusd_beta_60d deprecated/evicted - excluded from library corr reference)
B) batch N candidates (fresh families, low overlap with batches A-M):
   - vol-adjusted momentum acceleration: N_vol_adj_mom_accel_20x60
   - cross-sectional relative mom (60d vs median): N_mom_vs_median_60d
   - rolling market-correlation stability: N_roll_corr_stab_60d
   - return concentration (Herfindahl of |ret| weights): N_ret_herf_20d
   - skew of rolling 20d returns over 60d: N_skew_roll20_60d
   - down-market correlation only: N_dn_corr_mkt_60d
   - regime-cond downside beta (x VIX percentile): N_vix_cond_dnbeta_60d
   - US10Y-conditional rate beta: N_us10y_cond_beta_60d
   - relative momentum vs BTC: N_rel_mom_vs_btc_60d
   - volume skewness 20d: N_volume_skew_20d
   - DXY-up conditional beta: N_dxy_up_beta_60d
   - consecutive winning weeks (streak): N_week_win_streak_12w
   - trend R2 per unit vol: N_trend_r2_voladj_20d

Gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 at h=10 (15-asset universe, min_valid=8).
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
vol_panel = pd.DataFrame({a: panels[a]["volume"].astype(float) for a in closes.columns}).reindex(closes.index)
opens = pd.DataFrame({a: panels[a]["open"].astype(float) for a in closes.columns}).reindex(closes.index)
highs = pd.DataFrame({a: panels[a]["high"].astype(float) for a in closes.columns}).reindex(closes.index)
lows = pd.DataFrame({a: panels[a]["low"].astype(float) for a in closes.columns}).reindex(closes.index)
print(f"panels loaded {time.time()-t0:.1f}s | closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()}", flush=True)
print("last completed trading day:", closes.index.max().date(), flush=True)

H_ADM = 10
MIN_VALID = 8
GATE_IC, GATE_ICIR = 0.0070, 0.0840
mkt = rets.mean(axis=1)

# ---------------- vectorized rank-IC (Spearman via cross-sectional ranks) ----------------
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
    ssf = (cf ** 2).sum(axis=1)
    ssr = (cr ** 2).sum(axis=1)
    cov = (cf * cr).sum(axis=1)
    ic = cov / np.sqrt(ssf * ssr).replace(0, np.nan)
    ok = (nv >= min_valid) & (ssf > 1e-14) & (ssr > 1e-14) & ic.notna()
    return ic[ok].rename("ic")


def summarize_fast(ic_series: pd.Series):
    ic = float(ic_series.mean())
    sd = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else 0.0
    icir = ic / sd if sd > 0 else 0.0
    return {"ic": ic, "icir": icir, "ic_hit_ratio": float((ic_series > 0).mean()),
            "n_ic_dates": int(len(ic_series))}


def evaluate(tag, panel):
    fwd = forward_returns(closes, H_ADM)
    ics = rank_ic_series_fast(panel, fwd, MIN_VALID)
    m = summarize_fast(ics)
    m.update(coverage_metrics(panel, min_valid=MIN_VALID))
    m["turnover_10d_rank"] = turnover_rank(panel, 10)
    m["decay_ic_by_horizon"] = {}
    for h in (1, 2, 3, 5, 10, 20):
        fh = forward_returns(closes, h)
        ih = rank_ic_series_fast(panel, fh, MIN_VALID)
        if len(ih):
            m["decay_ic_by_horizon"][str(h)] = round(float(ih.mean()), 4)
    corr, key = max_library_corr(panel, LIBRARY)
    m["max_abs_library_correlation"] = corr
    m["max_corr_factor"] = key
    return m, ics


def rolling_beta(asset_ret, driver_ret, win=60, min_obs=40):
    beta = {}
    for a in asset_ret.columns:
        z = pd.concat([asset_ret[a].rename("a"), driver_ret.rename("m")], axis=1).dropna()
        cov = z["a"].rolling(win).cov(z["m"])
        var = z["m"].rolling(win).var()
        beta[a] = (cov / var).where(z["m"].rolling(win).count() >= min_obs)
    return pd.DataFrame(beta, index=asset_ret.index)


def rolling_corr_fast(a, b, win=60, min_obs=40):
    n = a.rolling(win).count()
    cov = (a * b).rolling(win).mean() - a.rolling(win).mean() * b.rolling(win).mean()
    den = a.rolling(win).std() * b.rolling(win).std()
    out = (cov / den.replace(0, np.nan)).where(n >= min_obs)
    return out


# ---------- ACTIVE library factor signal artifacts (exact library definitions) ----------
def lib_vol_price_corr_20():
    out = {}
    for a in closes.columns:
        out[a] = rolling_corr_fast(rets[a], vol_panel[a], 20, 15)
    return pd.DataFrame(out, index=rets.index)


def lib_dn_mkt_beta_60d():
    dn = mkt.where(mkt < 0)
    return rolling_beta(rets, dn, 60, 40)


def lib_rate_beta_cn10y_60d():
    cn = rets["CN10Y"]
    return rolling_beta(rets, cn, 60, 40)


LIBRARY = {
    "vol_price_corr_20": lib_vol_price_corr_20(),
    "dn_mkt_beta_60d": lib_dn_mkt_beta_60d(),
    "rate_beta_cn10y_60d": lib_rate_beta_cn10y_60d(),
}
print(f"library signals computed {time.time()-t0:.1f}s", flush=True)

# ---------- batch N candidates ----------
cands = {}
close_shift = closes.shift(1)

# vol-adjusted momentum acceleration
m20 = closes / closes.shift(20) - 1.0
m60 = closes / closes.shift(60) - 1.0
v20 = rets.rolling(20).std()
cands["N_vol_adj_mom_accel_20x60"] = (m20 - m60) / v20.replace(0, np.nan)

# cross-sectional relative momentum (60d vs median)
cands["N_mom_vs_median_60d"] = m60.sub(m60.median(axis=1), axis=0)

# rolling market-correlation stability: std of 20d corr-to-mkt over 60d
corr20 = pd.DataFrame({a: rolling_corr_fast(rets[a], mkt, 20, 15) for a in rets.columns}, index=rets.index)
cands["N_roll_corr_stab_60d"] = corr20.rolling(60).std()

# return concentration: Herfindahl of |daily ret| weights over 20d
absr = rets.abs()
num = (absr ** 2).rolling(20).sum()
den = absr.rolling(20).sum().replace(0, np.nan)
cands["N_ret_herf_20d"] = num / (den ** 2)

# skew of rolling 20d returns over 60d
r20 = rets.rolling(20).sum()
cands["N_skew_roll20_60d"] = r20.rolling(60).skew()

# down-market correlation only (60d)
dn = mkt.where(mkt < 0)
dn_corr = pd.DataFrame({a: rolling_corr_fast(rets[a], dn, 60, 40) for a in rets.columns}, index=rets.index)
cands["N_dn_corr_mkt_60d"] = dn_corr

# regime-conditional downside beta: dn_beta_60d x VIX percentile (VIX/VIX_250d median)
vix = panels["VIX"]["close"].astype(float)
vix_ratio = vix / vix.rolling(250).median()
vix_ratio_panel = pd.DataFrame({a: vix_ratio for a in rets.columns}, index=rets.index).reindex(rets.index)
dn_beta_60 = lib_dn_mkt_beta_60d()
cands["N_vix_cond_dnbeta_60d"] = dn_beta_60 * vix_ratio_panel

# US10Y-conditional rate beta: beta(asset, US10Y ret, 60) x sign(US10Y 20d mom)
us10y = rets["US10Y"]
us10y_beta = rolling_beta(rets, us10y, 60, 40)
us10y_mom20 = closes["US10Y"] / closes["US10Y"].shift(20) - 1.0
us10y_sign = pd.DataFrame({a: np.sign(us10y_mom20) for a in rets.columns}, index=rets.index).reindex(rets.index)
cands["N_us10y_cond_beta_60d"] = us10y_beta * us10y_sign

# relative momentum vs BTC (60d)
btc = closes["BTC"]
cands["N_rel_mom_vs_btc_60d"] = m60.sub(btc / btc.shift(60) - 1.0, axis=0)

# volume skewness 20d
cands["N_volume_skew_20d"] = vol_panel.rolling(20).skew()

# DXY-up conditional beta: beta(asset, DXY ret, 60) only on DXY-up days
dxy = panels["DXY"]["close"].pct_change() if "DXY" in panels else None
if dxy is not None:
    dxy_up = dxy.where(dxy > 0)
    cands["N_dxy_up_beta_60d"] = rolling_beta(rets, dxy_up, 60, 40)

# consecutive winning weeks (streak, capped at 12)
week_ret = rets.rolling(5).sum()
pos = (week_ret > 0).astype(float)
streak = pd.DataFrame(0.0, index=pos.index, columns=pos.columns)
for a in pos.columns:
    s = pos[a].copy()
    grp = (s != s.shift(1)).cumsum()
    cnt = s.groupby(grp).cumsum()
    streak[a] = cnt.where(s > 0, 0.0).clip(upper=12)
cands["N_week_win_streak_12w"] = streak

# trend R2 per unit vol (20d)
def trend_r2(series, win=20):
    x = np.arange(win)
    xm = x - x.mean()
    def _r2(y):
        if len(y) < win or np.std(y) == 0:
            return np.nan
        b = np.polyfit(x, y, 1)
        yhat = np.polyval(b, x)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return series.rolling(win).apply(_r2, raw=True)

tr2 = pd.DataFrame({a: trend_r2(closes[a], 20) for a in closes.columns}, index=closes.index)
cands["N_trend_r2_voladj_20d"] = tr2 / v20.replace(0, np.nan)

print(f"candidates defined {time.time()-t0:.1f}s ({len(cands)})", flush=True)

# ---------- evaluation ----------
results = {}
print("\n=== A) ACTIVE LIBRARY DRIFT (h=10) ===", flush=True)
for name, panel in LIBRARY.items():
    fwd = forward_returns(closes, H_ADM)
    ics = rank_ic_series_fast(panel, fwd, MIN_VALID)
    m = summarize_fast(ics)
    results[f"active_{name}"] = m
    line = f"{name}: full_ic={m['ic']:.4f} icir={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']}"
    for cut_name, cut in (("r250", closes.index[-250]), ("r500", closes.index[-500]), ("r750", closes.index[-750])):
        sub = ics[ics.index >= cut]
        if len(sub):
            icc = float(sub.mean())
            icirr = float(sub.mean() / sub.std(ddof=1)) if sub.std(ddof=1) > 0 else 0.0
            results[f"active_{name}"][f"ic_{cut_name}"] = round(icc, 4)
            results[f"active_{name}"][f"icir_{cut_name}"] = round(icirr, 4)
            line += f" | {cut_name}: ic={icc:.4f} icir={icirr:.4f}"
    print(line, flush=True)

print(f"\n=== B) BATCH N SCREEN ({len(cands)} candidates, h=10) ===", flush=True)
for tag, panel in cands.items():
    m, ics = evaluate(tag, panel)
    results[tag] = m
    for cut_name, cut in (("r250", closes.index[-250]), ("r500", closes.index[-500]), ("r750", closes.index[-750])):
        sub = ics[ics.index >= cut]
        if len(sub):
            results[tag][f"ic_{cut_name}"] = round(float(sub.mean()), 4)
            results[tag][f"icir_{cut_name}"] = round(float(sub.mean() / sub.std(ddof=1)), 4) if sub.std(ddof=1) > 0 else 0.0
    print(f"done {tag} {time.time()-t0:.1f}s", flush=True)

df = pd.DataFrame(results).T
df["pass"] = (df["ic"].abs() >= GATE_IC) & (df["icir"].abs() >= GATE_ICIR)
cols = ["ic", "icir", "ic_hit_ratio", "n_ic_dates", "ic_r250", "ic_r500", "ic_r750",
        "coverage_asset_days", "coverage_dates_ge8", "turnover_10d_rank",
        "max_abs_library_correlation", "max_corr_factor", "pass"]
print("\n=== FULL SCREEN (h=10, min_valid=8) ===")
print(df[cols].to_string(float_format=lambda x: f"{x:.4f}"))
print(f"\nPASSERS ({int(df['pass'].sum())}):", list(df.index[df["pass"]]), flush=True)

with open("scripts/_miner3_batchN_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print(f"\nelapsed {time.time()-t0:.1f}s", flush=True)
