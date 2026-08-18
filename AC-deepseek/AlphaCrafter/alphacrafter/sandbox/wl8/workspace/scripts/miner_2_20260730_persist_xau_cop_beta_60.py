"""miner_2 2026-07-30 — Persist xau_cop_beta_60 (Batch F passer).

Candidate passed the benchmark-wide admission gates on the 15-asset cross-asset
universe: |IC|=0.0518 >= 0.0070, |ICIR|=0.1309 >= 0.0840, admission horizon 10d,
max_abs_library_correlation=0.188 < 0.5 vs effective library (usdcny_beta_60).

This script recomputes the exact validation metrics + signal artifact and writes
factors/xau_cop_beta_60.json, then reads it back to verify.
"""
import sys
import json
import base64
import zlib
import io

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, validate_factor,
                                   IC_GATE, ICIR_GATE, artifact_b64)

CURRENT_DATE = "2026-07-30"
close, vol, open_, high, low = load_closes()
macro = {k: load_index(k) for k in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]}
print(f"Panel {close.index[0].date()}..{close.index[-1].date()}, {len(close)} rows x {close.shape[1]} assets")


def _beta(asset_ret, driver_ret, win):
    cov = asset_ret.rolling(win).cov(driver_ret)
    var = driver_ret.rolling(win).var()
    return cov / var


def f_xau_cop_beta_60(c, v, o, h, l, m, win=60):
    """60d beta of asset daily returns to XAU/COPPER ratio moves (gold vs industrial metals)."""
    xau = close["XAU"].reindex(c.index)
    cop = close["COPPER"].reindex(c.index)
    r = (xau / cop).pct_change()
    return _beta(c.pct_change(), r, win)


# ---- validation (same code path as batchF) ----
res = validate_factor(f_xau_cop_beta_60, close, vol, open_, high, low, macro)
panel = res["panel"]

# per-factor correlation vs effective library
def load_effective_library():
    lib = {}
    for f in ["usdcny_beta_60"]:
        try:
            d = json.load(open(f"factors/{f}.json"))
            if d.get("validation", {}).get("status") != "EFFECTIVE":
                continue
            raw = base64.b64decode(d["validation"]["signal_artifact"]["data"])
            p = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()), index_col=0, parse_dates=True)
            lib[f] = p
        except Exception as e:
            print(f"  [warn] library load {f}: {e}")
    return lib

lib = load_effective_library()
lib_corr_detail = {}
for fid, lp in lib.items():
    common = panel.index.intersection(lp.index)
    cols = [x for x in panel.columns if x in lp.columns]
    a = panel.loc[common, cols].values.ravel()
    b = lp.loc[common, cols].values.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    lib_corr_detail[fid] = round(float(np.corrcoef(a[m], b[m])[0, 1]), 4) if m.sum() > 200 else None
max_corr = max([abs(v) for v in lib_corr_detail.values() if v is not None] or [0.0])

# regime IC/ICIR breakdown
from factor_validation_lib import fwd_returns, ic_series
fr10 = fwd_returns(close, 10)
ic_main = ic_series(panel, fr10)
regimes = [
    ("2020-2021 COVID/recovery", "2020-01-01", "2021-12-31"),
    ("2022-2023 tightening/AI", "2022-01-01", "2023-12-31"),
    ("2024-2026-07 crypto/commodity", "2024-01-01", "2026-07-30"),
]
regime_ic = {}
for name, a, b in regimes:
    sub = ic_main.loc[a:b]
    if len(sub) >= 15:
        regime_ic[name] = [round(float(sub.mean()), 4), round(float(sub.mean() / sub.std()), 4), int(len(sub))]
    else:
        regime_ic[name] = None

metrics = {
    "ic": round(res["ic"], 6),
    "icir": round(res["icir"], 6),
    "ic_hit_ratio": round(res["ic_hit_ratio"], 4),
    "n_ic_dates": int(res["n_ic_dates"]),
    "coverage_asset_days": round(res["coverage_asset_days"], 4),
    "coverage_dates_ge8": round(res["coverage_dates_ge8"], 4),
    "turnover_10d_rank": round(res["turnover_10d_rank"], 4),
    "decay_ic_by_horizon": {k: round(v, 4) for k, v in res["decay_ic_by_horizon"].items()},
    "regime_ic_icir": regime_ic,
    "max_abs_library_correlation": round(max_corr, 4),
    "library_correlation_detail": lib_corr_detail,
}

doc = {
    "factor_id": "xau_cop_beta_60",
    "factor_name": "XAU/COPPER ratio beta (gold vs industrial metals sensitivity)",
    "version": "2026-07-30",
    "calculation": {
        "expression": "60d rolling beta(asset_daily_return, XAU/COPPER_ratio_daily_return)",
        "description": "Sensitivity of each asset's daily returns to moves in the gold/copper price ratio. "
                       "A high positive beta means the asset co-moves with gold relative to industrial metals "
                       "(defensive/real-asset tilt); a negative beta means it tracks the industrial-metals "
                       "(cyclical/risk-on) side. Unit-free and cross-sectionally comparable; captures the "
                       "risk-on/off commodity regime role of each instrument.",
    },
    "dependencies": ["close"],
    "parameters": {"window": 60, "admission_horizon": 10},
    "tags": ["beta", "cross-asset", "commodity", "ratio"],
    "validation": {
        "status": "EFFECTIVE",
        "period": "2020-01-01..2026-07-30",
        "last_validated": CURRENT_DATE,
        "admission_horizon": 10,
        "regime_notes": (
            "Validated on full 2020..2026-07 history (n=1465 IC dates, coverage 45.6% asset-days, "
            "61.5% dates have >=8 assets). Negative IC (high ratio-beta assets underperform at 10d horizon). "
            "Decay: |IC| rises with horizon (1d -0.014 -> 20d -0.071), strongest at 20d; admission at 10d "
            "(-0.052) is a compromise between signal strength and turnover (TO_10d_rank=0.90, low). "
            "Orthogonal to effective library: max |rho| = " + str(round(max_corr, 4)) + " vs [" + (" | ".join(lib.keys()) if lib else "none") + "]."
        ),
        "metrics": metrics,
        "signal_artifact": {
            "format": "csv+zlib+base64",
            "descrip": "Daily factor panel (dates x 15 assets) used for admission IC/ICIR computation; "
                       "recomputable by the post-Miner gate for pairwise signal rho.",
            "data": artifact_b64(panel),
        },
    },
    "expected_direction": -1,
    "benchmark_admission": {
        "contract": {"ic_threshold": IC_GATE, "icir_threshold": ICIR_GATE,
                     "correlation_threshold": 0.5, "library_capacity": 30, "active_top_k": 10},
        "selected_metrics": {
            "ic": metrics["ic"], "icir": metrics["icir"],
            "metric_path": "validation.metrics",
            "reported_max_abs_library_correlation": metrics["max_abs_library_correlation"],
            "correlation_path": "validation.metrics.max_abs_library_correlation",
            "quality": round(abs(metrics["ic"]) * abs(metrics["icir"]), 8),
        },
    },
}

path = "factors/xau_cop_beta_60.json"
with open(path, "w") as f:
    json.dump(doc, f, indent=1)
print("WROTE", path)

# ---- read-back verification ----
d2 = json.load(open(path))
assert d2["factor_id"] == "xau_cop_beta_60", "factor_id mismatch"
assert d2["validation"]["status"] == "EFFECTIVE", "status mismatch"
assert abs(d2["validation"]["metrics"]["ic"]) >= IC_GATE, "IC gate"
assert abs(d2["validation"]["metrics"]["icir"]) >= ICIR_GATE, "ICIR gate"
assert "signal_artifact" in d2["validation"], "missing artifact"
raw = base64.b64decode(d2["validation"]["signal_artifact"]["data"])
p2 = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()), index_col=0, parse_dates=True)
assert p2.shape == panel.shape, f"artifact shape {p2.shape} != {panel.shape}"
print("READBACK OK: id=%s status=%s ic=%s icir=%s libcorr=%s artifact=%s"
      % (d2["factor_id"], d2["validation"]["status"],
         metrics["ic"], metrics["icir"], metrics["max_abs_library_correlation"], p2.shape))
print("regime_ic_icir:", json.dumps(regime_ic))