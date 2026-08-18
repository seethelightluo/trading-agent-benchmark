"""miner_1 cycle 2026-11-24: screen novel candidate factors.

Extends the fastlib validation window to the current date (2026-11-24).
- MAX_VISIBLE = 2026-11-24 (current date; no lookahead beyond it)
- FACTOR_LAST = 2026-11-10 (h=10 forward returns fully inside visible data)
- Library = current 8 effective factors (recomputed deterministically)
Admission gates: |IC_h10| >= 0.007, |ICIR_h10| >= 0.084.
"""
import importlib.util, json, io, zlib, base64
from pathlib import Path
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("fastlib", "scripts/miner_1_fastlib.py")
fl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fl)

fl.MAX_VISIBLE = "2026-11-24"
fl.FACTOR_LAST = "2026-11-10"
WATCH = fl.WATCH
EPS = fl.EPS

panel = fl.load_panel()
print(f"panel: {panel.shape}, dates {panel.index.min().date()}..{panel.index.max().date()}", flush=True)
ohlc = fl.load_ohlc_volume()
macro = fl.load_macro()
rets = panel.pct_change()
mkt = panel.mean(axis=1).pct_change()


def library_signals_current(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Current 8-factor effective library (cycle-32 audit: kept=8)."""
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
    # corr_ew_60: mean pairwise rolling 60d correlation
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
    # kurt_20d_skip5
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

# 1) skew_20d_skip5: rolling 20d skewness of daily returns, skip 5 (risk asymmetry)
cands["skew_20d_skip5"] = rets.rolling(20, min_periods=12).skew().shift(5)

# 2) skew_60d: rolling 60d skewness (longer-horizon asymmetry)
cands["skew_60d"] = rets.rolling(60, min_periods=30).skew()

# 3) drawdown_60d: close/rolling_max(close,60)-1 (trend distance / reversal)
cands["drawdown_60d"] = panel / panel.rolling(60, min_periods=30).max() - 1.0

# 4) vol_ratio_10x60: 10d realized vol / 60d realized vol (vol term structure)
v10 = rets.rolling(10, min_periods=6).std()
v60 = rets.rolling(60, min_periods=30).std()
cands["vol_ratio_10x60"] = v10 / (v60 + EPS)

# 5) risk_on_beta_60d: rolling 60d beta of asset returns vs XAU/COPPER ratio returns
gcr = (panel["XAU"] / panel["COPPER"]).pct_change()
gcr_var = gcr.rolling(60, min_periods=30).var().replace(0, np.nan).reindex(panel.index)
cands["risk_on_beta_60d"] = rets.rolling(60, min_periods=30).cov(gcr).div(gcr_var.to_numpy(), axis=0)

# 6) usdjpy_beta_cond_60x20: beta(r, USDJPY_ret, 60) * (USDJPY 20d mom) (carry proxy)
jpy = macro["USDJPY"].pct_change()
jpy_var = jpy.rolling(60, min_periods=30).var().replace(0, np.nan).reindex(panel.index)
jb = rets.rolling(60, min_periods=30).cov(jpy).div(jpy_var.to_numpy(), axis=0)
jpy_mom = (macro["USDJPY"] / macro["USDJPY"].shift(20) - 1.0).reindex(panel.index)
cands["usdjpy_beta_cond_60x20"] = jb.mul(jpy_mom.to_numpy(), axis=0)

# 7) range_amplitude_20: mean((high-low)/close) over 20d
amp = {}
for s in WATCH:
    o = ohlc[s]
    amp[s] = ((o["high"] - o["low"]) / (o["close"] + EPS)).rolling(20, min_periods=10).mean()
cands["range_amplitude_20"] = pd.DataFrame(amp, index=panel.index).reindex(panel.index)

# 8) rsi_14: classic 14d RSI (contrarian oscillator)
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(n, min_periods=n).mean()
    dn = (-d.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = up / (dn + EPS)
    return 100.0 - 100.0 / (1.0 + rs)

cands["rsi_14"] = pd.DataFrame({s: rsi(panel[s]) for s in WATCH}, index=panel.index)

# 9) close_pos_range_20: mean((close-low)/(high-low)) over 20d (candle position)
cpr = {}
for s in WATCH:
    o = ohlc[s]
    rng = (o["high"] - o["low"]).replace(0, np.nan)
    cpr[s] = ((o["close"] - o["low"]) / rng).rolling(20, min_periods=10).mean()
cands["close_pos_range_20"] = pd.DataFrame(cpr, index=panel.index).reindex(panel.index)

# 10) cvar_20d: mean of worst 5% daily returns over 20d (tail risk)
def cvar20(x: pd.Series) -> pd.Series:
    return x.rolling(20, min_periods=12).apply(
        lambda w: float(np.mean(np.sort(w)[: max(1, int(0.05 * len(w)))])), raw=True)

cands["cvar_20d"] = pd.DataFrame({s: cvar20(rets[s]) for s in WATCH}, index=panel.index)

print("\n=== VALIDATION (window 2020-01-01..2026-11-10 factor, fwd through 2026-11-24) ===", flush=True)
def validate_fast_aligned(name, factor, panel, fwd, libs, fwd_rank_cache=None):
    """Same as fastlib.validate_fast but aligns fwd ranks to the truncated factor index."""
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
    print(f"{name:<24} IC10={r.get('ic_h10', float('nan')):+.4f} ICIR10={r.get('icir_h10', float('nan')):+.4f} "
          f"hit={r.get('hit_h10', float('nan')):.3f} cov={r.get('coverage_asset_days', float('nan')):.3f} "
          f"turn={r.get('turnover_10d_rank', float('nan')):.2f} maxcorr={r.get('max_abs_library_correlation')} "
          f"pass={g.get('pass')}", flush=True)

# save results for persistence step
with open("scripts/miner_1_20261124_screen_results.json", "w") as fh:
    slim = {k: {kk: vv for kk, vv in v.items() if kk != "library_corrs"} for k, v in results.items()}
    json.dump(slim, fh, indent=1, default=str)
print("saved scripts/miner_1_20261124_screen_results.json", flush=True)
