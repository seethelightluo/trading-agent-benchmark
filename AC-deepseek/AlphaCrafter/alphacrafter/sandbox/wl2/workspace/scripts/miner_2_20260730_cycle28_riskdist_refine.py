"""miner_2 cycle 28: refine return-distribution / downside-risk family.

Cycle 27 screen: skew_60 |IC|=0.0224 |ICIR|=0.0709 FAIL (close); downside_ratio_60
|IC|=0.0198 |ICIR|=0.0635 FAIL; maxdd_60 FAIL (lib corr 0.70). Refinements:
  - skew with other windows (30/90/120) and skip-1 (avoid overlap artifact)
  - gain_loss_ratio: mean(up ret)/|mean(down ret)| over 60d (asymmetry, not vol-normalized)
  - tail_ratio_60: p95(|ret|)/p50?  -> use (p95 - p5)/std as tail-fatness proxy (kurtosis-like)
  - cvar5_60: mean of worst 5% daily returns (downside tail depth)
  - drawdown_20 / drawdown_120 (shorter/longer window versions of maxdd)
  - downside_dev_ratio_30 / _90 (window variants)
Sign left raw; ensemble assigns direction from IC sign.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner2_lib import (load_close_panel, per_asset, validate_factor,
                        load_library_signals, report, forward_returns,
                        compute_ic, regime_breakdown)

panel = load_close_panel()
lib = load_library_signals(panel)
fwd_cache = {str(h): forward_returns(panel, h) for h in (1, 2, 3, 5, 10, 20)}
ret = panel.pct_change()

print("panel dates:", len(panel), "assets:", len(panel.columns))


def _gain_loss(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 20:
        return np.nan
    up = x[x > 0]
    dn = x[x < 0]
    if len(up) == 0 or len(dn) == 0:
        return np.nan
    return float(up.mean() / abs(dn.mean()))


def _tail_ratio(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 30:
        return np.nan
    sd = x.std()
    if sd <= 0:
        return np.nan
    return float((np.percentile(x, 95) - np.percentile(x, 5)) / sd)


def _cvar5(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 30:
        return np.nan
    return float(np.mean(np.sort(x)[: max(1, int(0.05 * len(x)))]))


def _maxdd(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return np.nan
    peak = np.maximum.accumulate(x)
    return float((x[-1] / np.max(x) - 1.0)) if np.max(x) > 0 else np.nan


def _downside_ratio(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 15:
        return np.nan
    mu = x.mean()
    dd = np.sqrt(np.mean((x[x < mu] - mu) ** 2)) if (x < mu).any() else 0.0
    sd = x.std()
    return dd / sd if sd > 0 else np.nan


cands = {}
# skew variants
for w in (30, 90, 120):
    cands[f"skew_{w}"] = ret.rolling(w, min_periods=max(15, w // 2)).skew()
# gain-loss asymmetry
cands["gain_loss_60"] = ret.rolling(60, min_periods=30).apply(_gain_loss, raw=True)
# tail fatness (kurtosis-like spread ratio)
cands["tail_ratio_60"] = ret.rolling(60, min_periods=30).apply(_tail_ratio, raw=True)
# downside tail depth (CVaR)
cands["cvar5_60"] = ret.rolling(60, min_periods=30).apply(_cvar5, raw=True)
# drawdown windows
for w in (20, 120):
    cands[f"drawdown_{w}"] = per_asset(panel, lambda s: s.rolling(w, min_periods=max(10, w // 4)).apply(_maxdd, raw=True))
# downside deviation ratio window variants
for w in (30, 90):
    cands[f"downside_dev_{w}"] = ret.rolling(w, min_periods=max(15, w // 2)).apply(_downside_ratio, raw=True)

results = {}
print("\n=== VALIDATION (admission horizon 10d; gate |IC|>=0.007 & |ICIR|>=0.084) ===")
for name, f in cands.items():
    m = validate_factor(f, panel, library=lib, fwd_cache=fwd_cache)
    p = report(name, m)
    results[name] = (m, p)

print("\n=== REGIME BREAKDOWNS for promising / borderline ===")
for name, (m, p) in results.items():
    if abs(m["ic"]) >= 0.007 or abs(m["icir"]) >= 0.06:
        ic_ser = compute_ic(cands[name], fwd_cache["10"]).dropna()
        rb = regime_breakdown(ic_ser)
        print(name, "PASS" if p else "FAIL", "| full ic", m["ic"], "icir", m["icir"],
              "| regimes", {k: (v["ic"], v["icir"]) for k, v in rb.items()},
              "| maxlibcorr", m.get("max_abs_library_correlation"))

print("\n=== SUMMARY ===")
for name, (m, p) in results.items():
    print(f"{name:22s} PASS={p} | ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} cov_asset={m['coverage_asset_days']:.3f} to10={m.get('turnover_10d_rank')} "
          f"maxlibcorr={m.get('max_abs_library_correlation')} decay={m['decay_ic_by_horizon']}")
