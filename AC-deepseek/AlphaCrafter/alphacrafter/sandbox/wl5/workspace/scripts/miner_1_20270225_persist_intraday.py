# -*- coding: utf-8 -*-
"""miner_1 2027-02-25 cycle: persist factors that passed the IC/ICIR gate.
tail_ratio_20  : ic=0.0325 icir=0.0967 (h=10, n=1440 dates)
dd_depth_20    : ic=0.0323 icir=0.0912 (h=10, n=808 dates)
Both are full-coverage (cov_a>=0.73) return-dynamics factors on the 15-asset universe.
Writes factors/<id>.json with signal_artifact, then reads back and verifies JSON.
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
import miner3_lib as L

L.LIB_FACTORS = ['trend_r2_30_signed', 'semi_down_ratio_20', 'mom_120d_skip5',
                 'mom_10d_skip5', 'time_under_water_120', 'vol_of_vol20x60',
                 'dxy_beta_60', 'WTI_BETA_60', 'vix_beta_cond_60x20', 'kurt_20']

VIS = '2027-02-25'
C, V, H, Lo, O = L.load_close_panel(4000)
mask = C.index < VIS
C, V, H, Lo, O = C[mask], V[mask], H[mask], Lo[mask], O[mask]
R = C.pct_change()
MP = lambda w: int(w * 0.5)

def _clean(a):
    a = np.asarray(a, dtype=float)
    return a[~np.isnan(a)]

def _apply_win(panel, w, fn):
    return panel.rolling(w, min_periods=MP(w)).apply(
        lambda a: fn(_clean(a)), raw=True)

def tail_ratio_20(w=20):
    q95 = _apply_win(R, w, lambda a: np.percentile(a, 95))
    q05 = _apply_win(R, w, lambda a: np.percentile(a, 5))
    return q95 / q05.abs().replace(0, np.nan)

def dd_depth_20(w=20):
    hi = C.rolling(w, min_periods=MP(w)).max()
    return (C - hi) / hi.replace(0, np.nan)

def recent_ic(fp, label):
    """IC over the most recent ~250d (2026-06 onward) as a freshness check."""
    s = L.rank_ic(fp, R.shift(-10))
    sub = s[s.index >= '2026-06-01']
    if len(sub) >= 60:
        return {"recent_ic": round(sub.mean(), 4),
                "recent_icir": round(sub.mean() / sub.std(), 4) if sub.std() > 0 else 0.0,
                "recent_n": int(len(sub))}
    return {"recent_ic": None, "recent_n": 0}

def persist(factor_id, factor_name, expression, description, deps, params, fp, extra_notes, tags):
    fp = fp.replace([np.inf, -np.inf], np.nan)
    s = L.rank_ic(fp, R.shift(-10))
    summ = L.summarize(s, 10, factor_id)
    summ['decay_ic_by_horizon'] = L.decay_analysis(fp, R)
    summ.update(L.coverage_turnover(fp, R, 10))
    rhos, maxrho = L.library_max_rho(fp)
    summ['library_rho_by_factor'] = rhos
    summ['max_abs_library_correlation'] = round(maxrho, 3)
    summ.update(recent_ic(fp, factor_id))
    artifact = L.build_artifact(fp)
    doc = {
        "factor_id": factor_id,
        "factor_name": factor_name,
        "version": "1.0.0",
        "calculation": {
            "expression": expression,
            "description": description
        },
        "dependencies": deps,
        "parameters": params,
        "expected_direction": 1,
        "validation": {
            "status": "EFFECTIVE",
            "period": "2020-01-01..2027-02-24",
            "admission_horizon": 10,
            "last_validated": "2027-02-25",
            "regime_notes": extra_notes,
            "metrics": summ,
            "signal_artifact": artifact
        },
        "tags": tags,
        "benchmark_admission": {
            "contract": {
                "ic_threshold": 0.007,
                "icir_threshold": 0.084,
                "correlation_threshold": 0.5,
                "library_capacity": 30,
                "active_top_k": 10
            },
            "selected_metrics": {
                "ic": summ["ic"],
                "icir": summ["icir"],
                "metric_path": "validation.metrics",
                "reported_max_abs_library_correlation": summ["max_abs_library_correlation"]
            }
        }
    }
    with open(f'factors/{factor_id}.json', 'w') as f:
        json.dump(doc, f, indent=1)
    print(f"WROTE factors/{factor_id}.json")
    return doc

# ---- tail_ratio_20 ----
tr = tail_ratio_20()
d1 = persist(
    'tail_ratio_20',
    '20-day Return Tail Ratio (q95/q05 asymmetry)',
    'rolling(q95(pct_change(close),20) / |q05(pct_change(close),20)|, min_periods=10)',
    'Ratio of the 95th to the 5th percentile of daily returns over the trailing 20 sessions '
    '(min_periods=10). Values > 1 indicate a fatter right tail (asymmetric upside bursts), '
    'values < 1 a fatter left tail. High right-tail-asymmetry assets outperform over the next '
    '10 days. Computed on the 15-asset tradable cross-asset universe; direction +1.',
    ["close"], {"window": 20, "min_periods": 10, "quantiles": [5, 95], "admission_horizon": 10},
    tr,
    "Cross-sectional rank IC vs 10-day forward returns on the 15-asset tradable universe. "
    "Positive IC in all regimes: 2020-2022 IC=0.0466 (ICIR=0.133), 2023-2024 IC=0.0225 "
    "(ICIR=0.070), 2025-2026 IC=0.0206 (ICIR=0.062); recent 2026-06..2027-02 IC positive. "
    "Signal peaks at 10d horizon (decay IC 0.0325 at h=10). Low correlation with the existing "
    "library (max |rho| 0.090), adding a distinct return-distribution asymmetry dimension.",
    ["tail-risk", "return-distribution", "cross-asset", "skew"]
)

# ---- dd_depth_20 ----
dd = dd_depth_20()
d2 = persist(
    'dd_depth_20',
    '20-day Drawdown Depth (vs rolling high)',
    '(close - rolling_max(close,20)) / rolling_max(close,20), min_periods=10',
    'Current drawdown depth relative to the trailing 20-session high (min_periods=10); '
    '0 = at highs, negative = below the 20d high. High values (near highs, shallow drawdown) '
    'outperform over the next 10 days: shallow-drawdown / fresh-high assets continue to lead. '
    'Complementary to time_under_water_120 (duration) with a shorter 20d depth horizon. '
    'Computed on the 15-asset tradable cross-asset universe; direction +1.',
    ["close"], {"window": 20, "min_periods": 10, "admission_horizon": 10},
    dd,
    "Cross-sectional rank IC vs 10-day forward returns on the 15-asset tradable universe. "
    "Strong in 2020-2022 (IC=0.0753, ICIR=0.201) and 2025-2026 (IC=0.0268, ICIR=0.081), "
    "weak-negative in 2023-2024 (IC=-0.0186) during the AI-led equity rally; recent "
    "2026-06..2027-02 IC positive. Max |rho| to library 0.096 - distinct from momentum/duration "
    "factors. Use as a shorter-horizon complement to time_under_water_120.",
    ["trend", "drawdown", "cross-asset", "mean-reversion"]
)

# ---- read-back verification ----
print("\n=== READ-BACK VERIFICATION ===")
for fid in ['tail_ratio_20', 'dd_depth_20']:
    with open(f'factors/{fid}.json') as f:
        doc = json.load(f)
    art = doc['validation']['signal_artifact']
    rec = L.decode_artifact(art)
    m = doc['validation']['metrics']
    ok = (doc['factor_id'] == fid and doc['validation']['status'] == 'EFFECTIVE'
          and abs(m['ic']) >= 0.0070 and abs(m['icir']) >= 0.0840
          and rec.shape[0] > 1000 and 'data' in art)
    print(f"{fid}: id={doc['factor_id']} status={doc['validation']['status']} "
          f"ic={m['ic']:.4f} icir={m['icir']:.4f} "
          f"max_rho={m['max_abs_library_correlation']:.3f} "
          f"artifact={rec.shape} valid={ok}")
    assert ok, f"VERIFICATION FAILED for {fid}"
print("ALL PERSISTED FACTORS VERIFIED OK")
