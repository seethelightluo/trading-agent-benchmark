"""miner_1 cycle17: persist vixbeta_60 (60d rolling beta of asset returns on VIX changes).
Passes admission gate: |IC|=0.0766>=0.007, |ICIR|=0.1848>=0.084, max_abs_library_correlation=0.1811<0.5.
"""
import sys, json, base64, zlib, io, datetime
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, validate_factor,
                                   max_library_corr, artifact_b64, print_result,
                                   ASSETS, IC_GATE, ICIR_GATE, CURRENT_DATE)

close, vol, open_, high, low = load_closes(CURRENT_DATE)
vix = load_index("VIX")
macro = {"vix": vix.pct_change()}

def vixbeta_60(close, vol, open_, high, low, macro, window=60):
    r = close.pct_change()
    m = macro["vix"].reindex(r.index)
    cov = r.rolling(window, min_periods=30).cov(m)
    var = m.rolling(window, min_periods=30).var()
    return (cov / var).replace([np.inf, -np.inf], np.nan)

res = validate_factor(vixbeta_60, close, vol, open_, high, low, macro,
                      horizons=(1, 2, 3, 5, 10, 20), admission_horizon=10)
print_result("vixbeta_60", res)
panel = res["panel"]

# library correlation vs ACTIVE library (usdcny_beta_60) and evicted informational factors
def load_panel(path):
    d = json.load(open(path))
    art = d["validation"]["signal_artifact"]
    p = pd.read_csv(io.StringIO(zlib.decompress(base64.b64decode(art["data"])).decode()),
                    index_col=0, parse_dates=True)
    p.index = pd.DatetimeIndex(p.index)
    return p

lib_panels = {}
for path, fid in [("factors/usdcny_beta_60.json", "usdcny_beta_60")]:
    try:
        lib_panels[fid] = load_panel(path)
    except Exception as e:
        print(f"[warn] {path}: {e}")

info_panels = {}
for path, fid in [("factors/evicted/mom_10d_skip5.json", "mom_10d_skip5"),
                  ("factors/evicted/vix_beta_cond_60x20.json", "vix_beta_cond_60x20"),
                  ("factors/evicted/yield_beta_cond_60x20.json", "yield_beta_cond_60x20"),
                  ("factors/evicted/mom_120d_skip5.json", "mom_120d_skip5")]:
    try:
        info_panels[fid] = load_panel(path)
    except Exception as e:
        print(f"[warn] {path}: {e}")

rho_active = max_library_corr(panel, lib_panels)
rho_info = max_library_corr(panel, info_panels)
print(f"max_rho_active_lib={rho_active:.4f}  max_rho_vs_evicted_info={rho_info:.4f}")

# regime IC on 10d horizon (union-panel approximate forward returns)
fr10 = close.pct_change(10).shift(-10)
regs = {}
for label, lo, hi in [("2020-2021 COVID/recovery", "2020-01-01", "2021-12-31"),
                      ("2022-2023 tightening/AI", "2022-01-01", "2023-12-31"),
                      ("2024-2026-07 crypto/commodity", "2024-01-01", "2026-07-30")]:
    sub = panel.loc[lo:hi]
    frs = fr10.loc[lo:hi]
    ics = []
    for dt in sub.index:
        x, y = sub.loc[dt], frs.loc[dt]
        m = x.notna() & y.notna()
        if m.sum() >= 8:
            ics.append(x[m].rank().corr(y[m].rank()))
    if ics:
        s = pd.Series(ics)
        regs[label] = [round(float(s.mean()), 4), round(float(s.mean() / s.std()), 4), len(s)]
print("regime IC:", regs)

ic, icir = res["ic"], res["icir"]
assert abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE, "gate fail"
assert rho_active < 0.5, "library correlation gate fail"

fid = "vixbeta_60"
doc = {
    "factor_id": fid,
    "factor_name": "60d beta vs VIX changes",
    "version": "1.0.0",
    "calculation": {
        "expression": "beta_60 = Cov(r_t, dVIX_t, 60d) / Var(dVIX_t, 60d), dVIX = daily pct change of VIX close",
        "description": "Rolling 60-trading-day beta of each asset's daily return on the daily percentage change of VIX (option-implied vol regime exposure). Negative/high-magnitude beta implies high sensitivity to vol spikes; used for defensive tilts (long low-VIX-beta assets, underweight high-VIX-beta assets in vol-sensitive regimes). Computed on each asset's own dense trading calendar and reindexed to the union panel."
    },
    "dependencies": ["close"],
    "parameters": {"window": 60, "horizon": 10, "min_periods": 30},
    "tags": ["beta", "volatility", "cross-asset", "risk"],
    "expected_direction": -1,
    "validation": {
        "status": "EFFECTIVE",
        "period": f"2020-01-01..{CURRENT_DATE.date()}",
        "last_validated": datetime.datetime.now().isoformat(timespec="seconds"),
        "admission_horizon": 10,
        "regime_notes": ("mixed/corrective regime; HIGH cross-sectional dispersion, LOW cross-asset correlation. "
                         "VIX 14.07 falling -14% 1M (low-vol regime); candidates validated on full 2020-2026 sample: "
                         "2020-21 IC -0.10 (vol-spike COVID), 2022-23 IC -0.08 (tightening), 2024-26-07 IC -0.05 (low-vol drift)."),
        "metrics": {
            "ic": round(ic, 4),
            "icir": round(icir, 4),
            "ic_hit_ratio": round(res["ic_hit_ratio"], 4),
            "n_ic_dates": int(res["n_ic_dates"]),
            "coverage_asset_days": res["coverage_asset_days"],
            "coverage_dates_ge8": res["coverage_dates_ge8"],
            "turnover_10d_rank": res["turnover_10d_rank"],
            "decay_ic_by_horizon": res["decay_ic_by_horizon"],
            "regime_ic_icir": regs,
            "max_abs_library_correlation": round(rho_active, 4),
            "library_correlation_detail": {"usdcny_beta_60": round(rho_active, 4)},
        },
        "signal_artifact": {
            "format": "base64:zlib:csv",
            "descrip": "factor value panel rows=date cols=asset (15-asset cross-asset universe)",
            "data": artifact_b64(panel),
        },
    },
    "benchmark_admission": {
        "contract": {"ic_threshold": IC_GATE, "icir_threshold": ICIR_GATE,
                     "correlation_threshold": 0.5, "library_capacity": 30, "active_top_k": 10},
        "selected_metrics": {"ic": round(ic, 4), "icir": round(icir, 4),
                             "metric_path": "validation.metrics",
                             "reported_max_abs_library_correlation": round(rho_active, 4),
                             "quality": round(abs(ic) * abs(icir), 8)},
        "admitted_at": datetime.datetime.now().isoformat(timespec="seconds"),
    },
}
path = f"factors/{fid}.json"
with open(path, "w") as fh:
    json.dump(doc, fh, indent=1, allow_nan=True)
print(f"PERSISTED {path}")