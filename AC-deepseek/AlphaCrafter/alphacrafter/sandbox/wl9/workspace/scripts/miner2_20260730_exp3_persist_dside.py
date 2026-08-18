"""Persist dside_ratio_21 (Exploration 3).

Passed admission gate at horizon 10 with NEGATIVE direction:
  IC -0.0415, ICIR -0.1263, |IC|>=0.0070, |ICIR|>=0.0840.
Interpretation: assets whose short-window downside semideviation is large
relative to upside semideviation (heavy left-tail/falling path) tend to
underperform over the next 10 days in this cross-asset universe
(volatility-drag / short-horizon reversal of distressed assets).

Construction: downside_ratio_21 = sqrt(mean(r_-^2, 21d)) / sqrt(mean(r_+^2, 21d))
where r_- = min(r,0), r_+ = max(r,0) on daily returns.
"""
import sys, json, os, io, base64, zlib, hashlib, datetime
sys.path.insert(0, "scripts")
from miner2_20260730_factorlib import load_panel, factor_panel, fwd_ret_panel, validate
import pandas as pd, numpy as np
from scipy.stats import spearmanr

P = load_panel()
R = P.pct_change()

def downside_ratio(s, w):
    neg = s.clip(upper=0)
    pos = s.clip(lower=0)
    d = np.sqrt((neg ** 2).rolling(w).mean())
    u = np.sqrt((pos ** 2).rolling(w).mean())
    return d / u.replace(0, np.nan)

fvals = factor_panel(R, lambda s: downside_ratio(s, 21))
fwd10 = fwd_ret_panel(P, 10)
res = validate(fvals, fwd10, label="dside_ratio_21", expected_dir=-1)
print("VALIDATION:", json.dumps(res))
assert res["passes"], "gate not passed - abort persistence"

# ---- max abs library correlation (self-reported provenance; gate recomputes) ----
def load_lib_factor(fid):
    d = json.load(open(f"factors/{fid}.json"))
    sa = d["validation"]["signal_artifact"]
    raw = zlib.decompress(base64.b64decode(sa["data"]))
    df = pd.read_csv(io.BytesIO(raw), index_col=0)
    df.index = pd.to_datetime(df.index)
    return df

LIB_IDS = ["mom_10d_skip5", "mom_120d_skip5", "vol_of_vol20x60", "vix_beta_cond_60x20"]
corrs = {}
for fid in LIB_IDS:
    try:
        lf = load_lib_factor(fid)
        common = fvals.index.intersection(lf.index)
        if len(common) < 100:
            corrs[fid] = None
            continue
        a = fvals.loc[common].stack()
        b = lf.loc[common].stack()
        m = a.notna() & b.notna()
        rho, _ = spearmanr(a[m], b[m])
        corrs[fid] = float(rho) if np.isfinite(rho) else None
    except Exception as e:
        corrs[fid] = f"ERR:{e}"
print("LIB_CORR:", json.dumps(corrs))
max_abs = max([abs(v) for v in corrs.values() if isinstance(v, float)], default=None)
print("MAX_ABS_LIB_CORR:", max_abs)

# ---- signal artifact (mirror library format base64:zlib:csv) ----
art = fvals.copy()
art = art.sort_index().sort_index(axis=1)
raw = art.to_csv().encode("utf-8")
comp = zlib.compress(raw)
b64 = base64.b64encode(comp).decode("ascii")
sha = hashlib.sha256(raw).hexdigest()
n_valid = int(art.notna().sum().sum())
signal_artifact = {
    "format": "base64:zlib:csv",
    "description": f"Factor signal panel: rows = dates, cols = assets. Shape {list(art.shape)}",
    "columns": list(art.columns),
    "shape": list(art.shape),
    "n_valid_values": n_valid,
    "sha256": sha,
    "data": b64,
}
print("ARTIFACT:", {k: v for k, v in signal_artifact.items() if k != "data"}, "| b64 len:", len(b64))

# ---- assembly ----
metrics = {
    "ic": res["ic"],
    "icir": res["icir"],
    "ic_hit_ratio": res["ic_hit_ratio"],
    "n_ic_dates": res["n_ic_dates"],
    "coverage_asset_days": res["coverage"],
    "coverage_dates_ge8": None,
    "turnover_10d_rank": res["turnover_10d_rank"],
    "decay_ic_by_horizon": res["decay_ic_by_horizon"],
    "max_abs_library_correlation": max_abs,
    "admission_direction": -1,
}
factor = {
    "factor_id": "dside_ratio_21",
    "factor_name": "Downside/upside semideviation ratio (21d)",
    "version": "1.0.0",
    "calculation": {
        "expression": "sqrt(mean(min(r,0)^2, 21)) / sqrt(mean(max(r,0)^2, 21)), r = pct_change(close)",
        "description": ("Ratio of downside to upside semideviation over the last 21 daily returns. "
                        "High values: asset path dominated by falling days with large downside semi-vol "
                        "(distress/hedging-demand regime). Validated NEGATIVE direction at 10d horizon: "
                        "high ratio -> lower forward return (short-horizon reversal / vol drag)."),
    },
    "dependencies": ["close"],
    "parameters": {"window": 21, "horizon": 10, "min_assets_for_ic": 8},
    "expected_direction": -1,
    "validation": {
        "status": "EFFECTIVE",
        "period": "2020-01-01..2026-07-29",
        "last_validated": "2026-07-30",
        "admission_horizon": 10,
        "regime_notes": ("Validated on 15-asset cross-asset universe 2020-01-01..2026-07-29 "
                         "across multiple regimes (COVID, 2022 tight, 2023-24 recovery, 2025-26 risk-on). "
                         "IC stable negative; decay strengthens by horizon 10-20."),
        "metrics": metrics,
        "signal_artifact": signal_artifact,
    },
    "tags": ["volatility", "skewness", "semideviation", "reversal"],
    "benchmark_admission": {
        "contract": {"ic_threshold": 0.007, "icir_threshold": 0.084,
                     "correlation_threshold": 0.5, "library_capacity": 30, "active_top_k": 10},
        "selected_metrics": {"ic": res["ic"], "icir": res["icir"],
                             "metric_path": "validation.metrics",
                             "reported_max_abs_library_correlation": max_abs},
        "admitted_at": datetime.datetime.now().isoformat(),
    },
}

out = "factors/dside_ratio_21.json"
with open(out, "w") as fh:
    json.dump(factor, fh)
print("WROTE", out, os.path.getsize(out), "bytes")

# ---- verify read-back ----
chk = json.load(open(out))
assert chk["factor_id"] == "dside_ratio_21"
assert chk["validation"]["status"] == "EFFECTIVE"
m = chk["validation"]["metrics"]
assert abs(m["ic"]) >= 0.0070 and abs(m["icir"]) >= 0.0840, "threshold check failed"
assert chk["validation"]["signal_artifact"]["format"] == "base64:zlib:csv"
rb = zlib.decompress(base64.b64decode(chk["validation"]["signal_artifact"]["data"]))
rd = pd.read_csv(io.BytesIO(rb), index_col=0)
assert rd.shape == art.shape and int(rd.notna().sum().sum()) == n_valid
print("READBACK OK: id=%s status=%s ic=%s icir=%s artifact_shape=%s n_valid=%s" % (
    chk["factor_id"], chk["validation"]["status"], m["ic"], m["icir"],
    chk["validation"]["signal_artifact"]["shape"], chk["validation"]["signal_artifact"]["n_valid_values"]))