"""miner_1 cycle 2026-12-08: screen novel candidate factors.

Extends the fastlib validation window to the current date (2026-12-08).
- MAX_VISIBLE = 2026-12-08 (current date; no lookahead beyond it)
- FACTOR_LAST = 2026-11-24 (h=10 forward returns fully inside visible data)
- Library = current 8 effective factors (recomputed deterministically)
Admission gates: |IC_h10| >= 0.007, |ICIR_h10| >= 0.084, max_abs_library_corr < 0.5.
New candidate families this cycle: trend-quality R2, return autocorrelation,
drawup excursion, vol expansion ratio, USDCNY/energy/safe-haven beta families,
risk-adjusted momentum, volume expansion, gap direction, bollinger position,
momentum persistence ratio.
"""
import importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("fastlib", "scripts/miner_1_fastlib.py")
fl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fl)

fl.MAX_VISIBLE = "2026-12-08"
fl.FACTOR_LAST = "2026-11-24"
WATCH = fl.WATCH
EPS = fl.EPS

panel = fl.load_panel()
print(f"panel: {panel.shape}, dates {panel.index.min().date()}..{panel.index.max().date()}", flush=True)
ohlc = fl.load_ohlc_volume()
macro = fl.load_macro()
rets = panel.pct_change()
mkt = panel.mean(axis=1).pct_change()


def library_signals_current(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Current 8-factor effective library (cycle-33 audit: kept=8)."""
    rets = panel.pct_change()
    mkt = panel.mean(axis=1).pct_change()
    v = rets.rolling(20, min_periods=10).std()
    out = {}
    out["max_ret_20d"] = rets.rolling(20, min_periods=10).max()
    dd = rets.clip(upper=0).rolling(20, min_periods=10).std()
    out["downside_vol_ratio_20"] = -(dd / (v + EPS))
    m20 = panel.shift(5) / panel.shift(25) - 1.0
    out["rel_mom_20d_skip5"] = m20.sub(m20.median(axis=1), axis=0)
    cov = rets.rolling(60, min_periods=30).cov(mkt)
    var = mkt.rolling(60, min_periods=30).var().replace(0, np.nan)
    out["beta_ew_60d"] = cov.div(var.to_numpy(), axis=0)
    idx = panel.index
    cs = {}
    for a in panel.columns:
        rows = []
        for b in panel.columns:
            if b == a:
                continue
            rows.append(rets[a].rolling(60, min_periods=30).corr(rets[b]))
        cs[a] = pd.concat(rows, axis=1).mean(axis=1)
    out["corr_ew_60"] = pd.DataFrame(cs, index=idx)
    out["kurt_20d_skip5"] = rets.rolling(20, min_periods=12).kurt().shift(5)
    macro = fl.load_macro()
    dxy = macro["DXY"].pct_change()
    dxy_var = dxy.rolling(60, min_periods=30).var().replace(0, np.nan).reindex(panel.index)
    db = rets.rolling(60, min_periods=30).cov(dxy).div(dxy_var.to_numpy(), axis=0)
    dxy_mom = (macro["DXY"] / macro["DXY"].shift(20) - 1.0).reindex(panel.index)
    out["dxy_beta_cond_60x20"] = -db.mul(dxy_mom.to_numpy(), axis=0)
    eur = macro["EURUSD"].pct_change()
    eur_var = eur.rolling(60, min_periods=30).var().replace(0, np.nan).reindex(panel.index)
    eb = rets.rolling(60, min_periods=30).cov(eur).div(eur_var.to_numpy(), axis=0)
    eur_mom = (macro["EURUSD"] / macro["EURUSD"].shift(20) - 1.0).reindex(panel.index)
    out["eurusd_beta_cond_60x20"] = eb.mul(eur_mom.to_numpy(), axis=0)
    return out


libs = library_signals_current(panel)
print("library recomputed:", {k: v.notna().sum().sum() for k, v in libs.items()}, flush=True)

fwd = {h: fl.fwd_returns(panel, h) for h in fl.HORIZONS}
fwd_rank_cache = {h: fwd[h].rank(axis=1).values.astype(float) for h in fl.HORIZONS}

# ---------------- candidate factors ----------------
cands = {}

# 1) trend_r2_20d_skip5: R2 of linear fit of log close over 20d (trend consistency), skip 5
def trend_r2(close: pd.Series, n: int = 20) -> pd.Series:
    x = np.arange(n, dtype=float)
    xm = x.mean()
    def _r2(w):
        if len(w) < n or np.any(~np.isfinite(w)):
            return np.nan
        y = np.log(np.asarray(w, dtype=float))
        ym = y.mean()
        sxx = ((x - xm) ** 2).sum()
        sxy = ((x - xm) * (y - ym)).sum()
        syy = ((y - ym) ** 2).sum()
        if sxx <= 0 or syy <= 0:
            return np.nan
        return (sxy * sxy) / (sxx * syy)
    return close.rolling(n, min_periods=n).apply(_r2, raw=True).shift(5)

cands["trend_r2_20d_skip5"] = pd.DataFrame({s: trend_r2(panel[s]) for s in WATCH}, index=panel.index)

# 2) autocorr_20d_skip5: lag-1 autocorrelation of daily returns over 20d, skip 5
def autocorr1(x: pd.Series, n: int = 20) -> pd.Series:
    return x.rolling(n, min_periods=12).apply(
        lambda w: float(pd.Series(w).autocorr(lag=1)) if len(w) >= 3 and np.std(w) > 0 else np.nan,
        raw=True).shift(5)

cands["autocorr_20d_skip5"] = pd.DataFrame({s: autocorr1(rets[s]) for s in WATCH}, index=panel.index)

# 3) drawup_20d: close/rolling_min(close,20)-1 (upside excursion / recovery strength)
cands["drawup_20d"] = panel / panel.rolling(20, min_periods=10).min() - 1.0

# 4) vol_expansion_20x60: 20d vol / 60d vol (vol term-structure expansion)
v20 = rets.rolling(20, min_periods=10).std()
v60 = rets.rolling(60, min_periods=30).std()
cands["vol_expansion_20x60"] = v20 / (v60 + EPS)

# 5) usdcny_beta_cond_60x20: beta(r, USDCNY_ret, 60) * (USDCNY 20d mom) (CNY carry/macro)
cny = macro["USDCNY"].pct_change()
cny_var = cny.rolling(60, min_periods=30).var().replace(0, np.nan).reindex(panel.index)
cb = rets.rolling(60, min_periods=30).cov(cny).div(cny_var.to_numpy(), axis=0)
cny_mom = (macro["USDCNY"] / macro["USDCNY"].shift(20) - 1.0).reindex(panel.index)
cands["usdcny_beta_cond_60x20"] = cb.mul(cny_mom.to_numpy(), axis=0)

# 6) wti_beta_60d: rolling 60d beta of asset returns vs WTI (energy beta)
wti = panel["WTI"].pct_change()
wti_var = wti.rolling(60, min_periods=30).var().replace(0, np.nan).reindex(panel.index)
cands["wti_beta_60d"] = rets.rolling(60, min_periods=30).cov(wti).div(wti_var.to_numpy(), axis=0)

# 7) xau_beta_60d: rolling 60d beta of asset returns vs XAU (safe-haven beta)
xau = panel["XAU"].pct_change()
xau_var = xau.rolling(60, min_periods=30).var().replace(0, np.nan).reindex(panel.index)
cands["xau_beta_60d"] = rets.rolling(60, min_periods=30).cov(xau).div(xau_var.to_numpy(), axis=0)

# 8) sharpe_ratio_60d_skip5: mean/std of 60d daily returns, skip 5 (risk-adjusted momentum)
mu = rets.rolling(60, min_periods=30).mean()
sd = rets.rolling(60, min_periods=30).std()
cands["sharpe_ratio_60d_skip5"] = (mu / (sd + EPS)).shift(5)

# 9) volume_expansion_20d: 5d avg volume / 60d avg volume (attention proxy)
vol_exp = {}
for s in WATCH:
    vdf = ohlc[s]["volume"].astype(float)
    vol_exp[s] = vdf.rolling(5, min_periods=3).mean() / (vdf.rolling(60, min_periods=30).mean() + EPS)
cands["volume_expansion_20d"] = pd.DataFrame(vol_exp, index=panel.index).reindex(panel.index)

# 10) gap_direction_20d: mean((open - prev_close)/prev_close) over 20d (gap continuation)
gap = {}
for s in WATCH:
    o = ohlc[s]
    g = o["open"] / o["close"].shift(1) - 1.0
    gap[s] = g.rolling(20, min_periods=10).mean()
cands["gap_direction_20d"] = pd.DataFrame(gap, index=panel.index).reindex(panel.index)

# 11) bollinger_position_20x2: (close - sma20)/(2*std20) (mean-reversion distance)
sma20 = panel.rolling(20, min_periods=10).mean()
std20 = rets.rolling(20, min_periods=10).std()
cands["bollinger_position_20x2"] = (panel - sma20) / (2 * std20 * panel + EPS)

# 12) mom_persistence_20x60: 20d mom (skip5) / 60d mom (skip5) abs-ratio (momentum persistence)
m20 = panel.shift(5) / panel.shift(25) - 1.0
m60 = panel.shift(5) / panel.shift(65) - 1.0
cands["mom_persistence_20x60"] = m20 / (m60.abs() + EPS)

print("\n=== VALIDATION (window 2020-01-01..2026-11-24 factor, fwd through 2026-12-08) ===", flush=True)


def validate_fast_aligned(name, factor, panel, fwd, libs, fwd_rank_cache=None):
    factor = factor.reindex(panel.index).loc[:fl.FACTOR_LAST]
    n_valid = int(factor.notna().sum().sum())
    if n_valid < 100:
        return {"name": name, "factor_rows": len(factor), "n_assets": panel.shape[1],
                "admission_gate": {"pass": False}, "reason": "insufficient_data", "n_valid": n_valid}
    res = {"name": name, "factor_rows": int(len(factor)), "n_assets": panel.shape[1]}
    ic_by_h = {}
    for h in fl.HORIZONS:
        F = factor.rank(axis=1).values.astype(float)
        if fwd_rank_cache is not None and h in fwd_rank_cache:
            R = pd.DataFrame(fwd_rank_cache[h], index=panel.index).reindex(factor.index).values.astype(float)
        else:
            R = fwd[h].reindex(factor.index).rank(axis=1).values.astype(float)
        ic_by_h[h] = pd.Series(fl.row_pearson(F, R), index=factor.index)
    ic10 = ic_by_h[10]
    direction = float(np.sign(ic10.mean())) if np.isfinite(ic10.mean()) and ic10.mean() != 0 else 1.0
    for h in fl.HORIZONS:
        ic = ic_by_h[h] * direction
        res[f"ic_h{h}"] = float(ic.mean())
        res[f"icir_h{h}"] = float(ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
        res[f"hit_h{h}"] = float((ic > 0).mean())
        res[f"n_dates_h{h}"] = int(len(ic))
    res["direction"] = direction
    res["raw_ic_h10"] = float(ic10.mean())
    valid = factor.notna()
    res["coverage_asset_days"] = float(valid.mean().mean())
    res["coverage_dates_ge8"] = float((valid.sum(axis=1) >= fl.MIN_ASSETS).mean())
    res["turnover_10d_rank"] = fl.turnover_10d_rank_fast(factor)
    max_corr, per = fl.lib_corr_fast(factor, libs)
    res["max_abs_library_correlation"] = max_corr
    res["library_corrs"] = per
    res["decay_ic_by_horizon"] = {str(h): round(res[f"ic_h{h}"], 4) for h in fl.HORIZONS}
    gate_ic = abs(res["ic_h10"]) >= fl.ADMISSION["ic"]
    gate_icir = abs(res["icir_h10"]) >= fl.ADMISSION["icir"]
    gate_corr = (res["max_abs_library_correlation"] is None
                 or not np.isfinite(res["max_abs_library_correlation"])
                 or res["max_abs_library_correlation"] < fl.ADMISSION["corr"])
    res["admission_gate"] = {"ic_pass": bool(gate_ic), "icir_pass": bool(gate_icir),
                             "corr_pass": bool(gate_corr),
                             "pass": bool(gate_ic and gate_icir and gate_corr)}
    flag = "PASS" if res["admission_gate"]["pass"] else "FAIL"
    print(f"  {name:<28} h10 IC={res['ic_h10']:+.4f} ICIR={res['icir_h10']:+.4f} "
          f"hit={res['hit_h10']:.3f} cov={res['coverage_asset_days']:.3f} "
          f"turn={res['turnover_10d_rank']:.2f} maxcorr={res['max_abs_library_correlation']} "
          f"-> {flag}", flush=True)
    return res


results = {}
for name, f in cands.items():
    res = validate_fast_aligned(name, f, panel, fwd, libs, fwd_rank_cache)
    results[name] = res

print("\n=== SUMMARY ===", flush=True)
for name, r in results.items():
    g = r.get("admission_gate", {})
    print(f"{name:<28} IC10={r.get('ic_h10', float('nan')):+.4f} ICIR10={r.get('icir_h10', float('nan')):+.4f} "
          f"hit={r.get('hit_h10', float('nan')):.3f} cov={r.get('coverage_asset_days', float('nan')):.3f} "
          f"turn={r.get('turnover_10d_rank', float('nan')):.2f} maxcorr={r.get('max_abs_library_correlation')} "
          f"pass={g.get('pass')}", flush=True)

with open("scripts/miner_1_20261208_screen_results.json", "w") as fh:
    slim = {k: {kk: vv for kk, vv in v.items() if kk != "library_corrs"} for k, v in results.items()}
    json.dump(slim, fh, indent=1, default=str)
print("saved scripts/miner_1_20261208_screen_results.json", flush=True)
