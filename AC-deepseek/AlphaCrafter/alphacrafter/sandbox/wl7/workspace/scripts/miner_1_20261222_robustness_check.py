"""miner_1 2026-12-22: robustness re-validation of the two gate-passing candidates
from the 2026-12-08 screen (autocorr_20d_skip5, vol_expansion_20x60).

Focus:
- Re-validate on the current visible window (MAX_VISIBLE=2026-12-22, h=10 forward
  fully inside visible data requires FACTOR_LAST=2026-12-08).
- Sub-period / regime IC stability (year-by-year + recent 250d).
- Asset-level coverage: per-asset valid fraction; number of dates with >=8 valid.
- Anomaly check: autocorr hit ratio ~0.018 despite positive mean IC -> examine
  distribution of daily IC and whether a few extreme dates drive the mean.
- Library correlation vs the CURRENT 7-factor ensemble library (not 8).

Admission gates: |IC_h10|>=0.007, |ICIR_h10|>=0.084, max_abs_library_corr<0.5.
"""
import importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("fastlib", "scripts/miner_1_fastlib.py")
fl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fl)

fl.MAX_VISIBLE = "2026-12-22"
fl.FACTOR_LAST = "2026-12-08"
WATCH = fl.WATCH
EPS = fl.EPS

panel = fl.load_panel()
print(f"panel: {panel.shape}, dates {panel.index.min().date()}..{panel.index.max().date()}", flush=True)
rets = panel.pct_change()
mkt = panel.mean(axis=1).pct_change()

# ---- current 7-factor ensemble library (matches factor_ensemble.json) ----
def library_current(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
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
    eur = macro["EURUSD"].pct_change()
    eur_var = eur.rolling(60, min_periods=30).var().replace(0, np.nan).reindex(panel.index)
    eb = rets.rolling(60, min_periods=30).cov(eur).div(eur_var.to_numpy(), axis=0)
    eur_mom = (macro["EURUSD"] / macro["EURUSD"].shift(20) - 1.0).reindex(panel.index)
    out["eurusd_beta_cond_60x20"] = eb.mul(eur_mom.to_numpy(), axis=0)
    return out

libs = library_current(panel)
print("library recomputed:", {k: v.notna().sum().sum() for k, v in libs.items()}, flush=True)

fwd = {h: fl.fwd_returns(panel, h) for h in fl.HORIZONS}
fwd_rank_cache = {h: fwd[h].rank(axis=1).values.astype(float) for h in fl.HORIZONS}

# ---- candidate factor definitions (identical to 2026-12-08 screen) ----
def autocorr1(x: pd.Series, n: int = 20) -> pd.Series:
    return x.rolling(n, min_periods=12).apply(
        lambda w: float(pd.Series(w).autocorr(lag=1)) if len(w) >= 3 and np.std(w) > 0 else np.nan,
        raw=True).shift(5)

v20 = rets.rolling(20, min_periods=10).std()
v60 = rets.rolling(60, min_periods=30).std()

cands = {
    "autocorr_20d_skip5": pd.DataFrame({s: autocorr1(rets[s]) for s in WATCH}, index=panel.index),
    "vol_expansion_20x60": v20 / (v60 + EPS),
}

def detailed_validate(name, factor):
    factor = factor.reindex(panel.index).loc[:fl.FACTOR_LAST]
    n_valid = int(factor.notna().sum().sum())
    res = {"name": name, "factor_rows": len(factor), "n_assets": panel.shape[1], "n_valid": n_valid}
    if n_valid < 100:
        print(f"  {name}: INSUFFICIENT DATA n_valid={n_valid}")
        return res
    ic_by_h = {}
    for h in fl.HORIZONS:
        F = factor.rank(axis=1).values.astype(float)
        R = pd.DataFrame(fwd_rank_cache[h], index=panel.index).reindex(factor.index).values.astype(float)
        ic_by_h[h] = pd.Series(fl.row_pearson(F, R), index=factor.index)
    ic10 = ic_by_h[10]
    direction = float(np.sign(ic10.mean())) if np.isfinite(ic10.mean()) and ic10.mean() != 0 else 1.0
    for h in fl.HORIZONS:
        ic = ic_by_h[h] * direction
        res[f"ic_h{h}"] = float(ic.mean())
        res[f"icir_h{h}"] = float(ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan")
        res[f"hit_h{h}"] = float((ic > 0).mean())
    res["direction"] = direction
    res["raw_ic_h10"] = float(ic10.mean())
    valid = factor.notna()
    res["coverage_asset_days"] = float(valid.mean().mean())
    res["coverage_dates_ge8"] = float((valid.sum(axis=1) >= fl.MIN_ASSETS).mean())
    res["n_dates_ge8"] = int((valid.sum(axis=1) >= fl.MIN_ASSETS).sum())
    res["turnover_10d_rank"] = fl.turnover_10d_rank_fast(factor)
    max_corr, per = fl.lib_corr_fast(factor, libs)
    res["max_abs_library_correlation"] = max_corr
    res["library_corrs"] = per
    res["decay_ic_by_horizon"] = {str(h): round(res[f"ic_h{h}"], 4) for h in fl.HORIZONS}

    # sub-period stability
    ic_pos = ic10 * direction
    yr = {}
    for y in sorted(set(ic_pos.index.year)):
        sub = ic_pos[ic_pos.index.year == y].dropna()
        if len(sub) > 5:
            yr[str(y)] = {"ic": round(float(sub.mean()), 4), "icir": round(float(sub.mean() / sub.std()), 3) if sub.std() > 0 else None, "n": int(len(sub))}
    res["ic_by_year"] = yr
    last250 = ic_pos.dropna().tail(250)
    res["ic_last250"] = round(float(last250.mean()), 4) if len(last250) > 5 else None
    res["icir_last250"] = round(float(last250.mean() / last250.std()), 3) if len(last250) > 5 and last250.std() > 0 else None
    res["n_last250"] = int(len(last250))

    # asset-level coverage
    cov_by_asset = valid.mean()
    res["asset_coverage_min"] = float(cov_by_asset.min())
    res["asset_coverage_max"] = float(cov_by_asset.max())
    res["assets_cov_lt_50pct"] = [a for a in valid.columns if cov_by_asset[a] < 0.5]
    res["assets_cov_lt_10pct"] = [a for a in valid.columns if cov_by_asset[a] < 0.1]

    # IC distribution diagnostics (h10)
    ic_vals = (ic10 * direction).dropna()
    res["ic_h10_n_dates_valid"] = int(len(ic_vals))
    res["ic_h10_median"] = float(ic_vals.median())
    res["ic_h10_p90"] = float(np.percentile(ic_vals, 90))
    res["ic_h10_max"] = float(ic_vals.max())
    res["ic_h10_skew"] = float(ic_vals.skew()) if len(ic_vals) > 3 else None
    # dates with largest |IC|
    top = ic_vals.abs().nlargest(5)
    res["top5_abs_ic_dates"] = [(str(d.date()), round(float(x), 4)) for d, x in top.items()]

    gate_ic = abs(res["ic_h10"]) >= fl.ADMISSION["ic"]
    gate_icir = abs(res["icir_h10"]) >= fl.ADMISSION["icir"]
    gate_corr = (res["max_abs_library_correlation"] is None
                 or not np.isfinite(res["max_abs_library_correlation"])
                 or res["max_abs_library_correlation"] < fl.ADMISSION["corr"])
    res["admission_gate"] = {"ic_pass": bool(gate_ic), "icir_pass": bool(gate_icir),
                             "corr_pass": bool(gate_corr),
                             "pass": bool(gate_ic and gate_icir and gate_corr)}
    flag = "PASS" if res["admission_gate"]["pass"] else "FAIL"
    print(f"  {name:<26} h10 IC={res['ic_h10']:+.4f} ICIR={res['icir_h10']:+.4f} "
          f"hit={res['hit_h10']:.3f} covAD={res['coverage_asset_days']:.3f} covD8={res['coverage_dates_ge8']:.3f} "
          f"nD8={res['n_dates_ge8']} turn={res['turnover_10d_rank']:.2f} maxcorr={res['max_abs_library_correlation']} "
          f"-> {flag}", flush=True)
    return res

print("\n=== DETAILED VALIDATION (window through 2026-12-08) ===", flush=True)
results = {}
for name, f in cands.items():
    results[name] = detailed_validate(name, f)

print("\n=== SUB-PERIOD DETAIL ===", flush=True)
for name, r in results.items():
    print(f"\n[{name}] direction={r.get('direction')} raw_ic_h10={r.get('raw_ic_h10'):+.4f} "
          f"ic_median={r.get('ic_h10_median')} p90={r.get('ic_h10_p90')} max={r.get('ic_h10_max')} skew={r.get('ic_h10_skew')}")
    print(f"  coverage: min_asset={r.get('asset_coverage_min'):.3f} max_asset={r.get('asset_coverage_max'):.3f} "
          f"assets<50%={r.get('assets_cov_lt_50pct')} assets<10%={r.get('assets_cov_lt_10pct')}")
    print(f"  ic_by_year: {json.dumps(r.get('ic_by_year', {}), indent=1)}")
    print(f"  last250: ic={r.get('ic_last250')} icir={r.get('icir_last250')} n={r.get('n_last250')}")
    print(f"  top5_abs_ic_dates: {r.get('top5_abs_ic_dates')}")

with open("scripts/miner_1_20261222_robustness_results.json", "w") as fh:
    json.dump(results, fh, indent=1, default=str)
print("\nsaved scripts/miner_1_20261222_robustness_results.json", flush=True)
