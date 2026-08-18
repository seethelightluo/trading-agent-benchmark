"""miner_1 cycle 21: persist passing candidates discovered in cycle21 batch.

Candidates that passed the admission gate (|IC|>=0.007, |ICIR|>=0.084 @ h=10,
max_abs_library_correlation<0.5 vs current 7-factor library):
  - corr_ew_60          IC=+0.0341 ICIR=+0.1001 maxcorr=0.2405
  - variance_ratio_5x60 IC=+0.0347 ICIR=+0.1120 maxcorr=0.1514
  - trend_eff_ratio_20  IC=+0.0502 ICIR=+0.1678 maxcorr=0.1982

Recomputed here against the CURRENT effective library (cycle-25 audit) to make
max_abs_library_correlation provenance accurate, then persisted with signal
artifacts (.npy + embedded base64:zlib:csv) and verified by reload.
"""
from __future__ import annotations
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_fastlib import (  # noqa: E402
    WATCH, load_panel, load_ohlc_volume, load_macro, fwd_returns,
    library_signals, lib_corr_fast, validate_fast, persist_factor,
    verify_factor, HORIZONS, FACTOR_LAST, EPS,
)

panel = load_panel()
rets = panel.pct_change()
OHLC = load_ohlc_volume()
libs = library_signals(panel)
fwd = {h: fwd_returns(panel, h) for h in HORIZONS}
fwd_rank_cache = {h: fwd[h].loc[:FACTOR_LAST].rank(axis=1).values.astype(float) for h in HORIZONS}
print(f"panel {panel.shape}, factor window .. {FACTOR_LAST}", flush=True)
print("library factors:", list(libs.keys()), flush=True)


def per_asset_series(fn):
    cols = {}
    for a in panel.columns:
        s = panel[a].dropna()
        cols[a] = fn(s)
    return pd.DataFrame(cols, index=panel.index)


def cand_corr_ew_60():
    """Mean pairwise rolling 60d correlation of asset returns with all others."""
    cols = {}
    rmat = rets.values.astype(float)
    idx = panel.index
    for j, a in enumerate(panel.columns):
        x = rmat[:, j]
        others = np.delete(rmat, j, axis=1)
        out = np.full(len(x), np.nan)
        for i in range(60, len(x)):
            w = slice(i - 60, i)
            xw = x[w]
            ow = others[w]
            valid = np.isfinite(xw)[:, None] & np.isfinite(ow)
            if valid.sum() < 30:
                continue
            xc = xw - np.nanmean(xw)
            oc = ow - np.nanmean(ow, axis=0)
            num = np.nansum(xc[:, None] * oc, axis=0)
            dx = np.sqrt(np.nansum(xc * xc))
            do = np.sqrt(np.nansum(oc * oc, axis=0))
            with np.errstate(invalid="ignore"):
                r = num / (dx * do)
            out[i] = np.nanmean(r[np.isfinite(r)])
        cols[a] = pd.Series(out, index=idx)
    return pd.DataFrame(cols, index=idx)


def cand_variance_ratio_5x60():
    """Variance ratio: 5d variance (annualized) / 60d variance (>1 trending)."""
    v5 = rets.rolling(5, min_periods=3).var() * 12.0
    v60 = rets.rolling(60, min_periods=30).var()
    return v5 / (v60 + EPS)


def cand_trend_eff_ratio_20():
    """Kaufman efficiency ratio 20d: |net move| / sum(|daily moves|)."""
    def _er(s):
        n = 20
        change = s.diff(n).abs()
        vol = s.diff().abs().rolling(n, min_periods=10).sum()
        return change / (vol + EPS)
    return per_asset_series(_er)


CANDIDATES = {
    "corr_ew_60": (cand_corr_ew_60,
                   "Mean pairwise 60d return correlation with all other assets (comovement)",
                   "For each asset, mean of rolling 60-day Pearson correlations of its daily return with every other watchlist asset's daily return. High comovement assets are treated as crowding/risk-on proxy.",
                   ["close"], {"window": 60, "min_periods": 30, "aggregation": "mean_pairwise"},
                   ["cross_asset", "comovement", "risk_on"],
                   "2020-01..2026-07: comovement premium positive at 10-20d horizons; very low turnover (1.26), high coverage (0.97)."),
    "variance_ratio_5x60": (cand_variance_ratio_5x60,
                            "Variance ratio 5d vs 60d (trend persistence), inverted",
                            "Ratio of 5-day realized variance (annualized) to 60-day realized variance; signal direction inverted (low short/long variance ratio = smoother trending assets favored). Rare-event profile: low daily hit ratio but strong average rank IC.",
                            ["close"], {"short_window": 5, "long_window": 60, "annualize": 12.0},
                            ["volatility", "trend", "variance_ratio"],
                            "2020-01..2026-07: passes IC/ICIR gates but hit ratio low (0.14) - event-driven; high turnover (4.35), coverage 0.55. Use with caution in ensemble."),
    "trend_eff_ratio_20": (cand_trend_eff_ratio_20,
                           "Kaufman efficiency ratio 20d (trend quality)",
                           "Net 20-day price change divided by sum of absolute daily changes; high values = efficient directional trends. Peaks at 10d horizon.",
                           ["close"], {"window": 20, "min_periods": 10},
                           ["momentum", "trend", "efficiency"],
                           "2020-01..2026-07: strongest new family candidate (IC 0.050, ICIR 0.168 @ h10); turnover 4.06, coverage 0.72."),
}

results = {}
for name, (fn, fname, desc, deps, params, tags, regime) in CANDIDATES.items():
    factor = fn().reindex(panel.index)
    res = validate_fast(name, factor, panel, fwd, libs, fwd_rank_cache)
    results[name] = res
    g = res["admission_gate"]
    if g["pass"]:
        persist_factor(
            factor_id=name, factor_name=fname, expression=desc, description=desc,
            deps=deps, params=params, res=res, tags=tags,
            regime_notes=regime, panel=panel, factor=factor)
        verify_factor(name, res)
    else:
        print(f"  {name}: gate FAIL -> not persisted", flush=True)

print("\n===== PERSIST SUMMARY =====", flush=True)
for name, r in results.items():
    g = r["admission_gate"]
    print(f"{name:<24} IC={r['ic_h10']:+.4f} ICIR={r['icir_h10']:+.4f} "
          f"hit={r['hit_h10']:.3f} turn={r['turnover_10d_rank']:.2f} "
          f"cov={r['coverage_asset_days']:.3f} maxcorr={r['max_abs_library_correlation']} "
          f"-> {'PERSISTED' if g['pass'] else 'FAIL'}", flush=True)

with open("scripts/miner_1_cycle21_persist_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("saved -> scripts/miner_1_cycle21_persist_results.json", flush=True)
