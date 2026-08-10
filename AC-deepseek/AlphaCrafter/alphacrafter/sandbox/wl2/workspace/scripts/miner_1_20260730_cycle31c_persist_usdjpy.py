"""miner_1 cycle 31c: persist usdjpy_beta_cond_120x60 and update ensemble to 4 factors.

Verified in cycle31b exploration:
  usdjpy_beta_cond_120x60: IC=0.0456, ICIR=0.1377, maxlibcorr=0.0839,
  all subperiods positive (2020-21 +0.0506, 2022 +0.0339, 2023-24 +0.0150,
  2025-26 +0.0875 icir +0.303), turnover 0.14 -> PASS all gates.
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (TRADABLES, VISIBLE_THROUGH, load_asset, load_panel,
                         macro_series, per_asset, forward_returns, compute_ic,
                         validate_factor)

# ---------------------------------------------------------------------------
# 1. Recompute signal and validate
# ---------------------------------------------------------------------------
close_panel = load_panel()

def beta_cond(asset_close, driver_close, w=60, m=20, minp_frac=0.5):
    dcs = driver_close.reindex(asset_close.index).ffill()
    ar = asset_close.pct_change()
    dr = dcs.pct_change()
    df = pd.concat([ar.rename("a"), dr.rename("d")], axis=1).dropna()
    minp = max(int(w * minp_frac), 15)
    cov = df["a"].rolling(w, min_periods=minp).cov(df["d"])
    var = df["d"].rolling(w, min_periods=minp).var()
    beta = cov / var
    mom = dcs / dcs.shift(m) - 1.0
    return beta * mom.reindex(beta.index)

usdjpy = macro_series("USDJPY")
sig = per_asset(close_panel, beta_cond, usdjpy, 120, 60)

# library signals
def mom20_volproxy60(s):
    mom = s.shift(5) / s.shift(25) - 1.0
    proxy = s.shift(5) / s.shift(65) - 1.0
    return mom / (1.0 + proxy.abs())

def calmness_20(s):
    r = s.pct_change()
    v = r.rolling(20).std()
    return r.abs().rolling(20).apply(
        lambda x: float((np.abs(x) < 0.5 * np.nanstd(x)).mean()) if len(x) >= 10 else np.nan,
        raw=True)

def intraday_drift_20(close_s):
    frames_a = {a: load_asset(a) for a in TRADABLES}
    op = pd.Series(frames_a[close_s.name]["open"].astype(float).values,
                   index=pd.to_datetime(frames_a[close_s.name]["date"]), name=close_s.name)
    return (close_s / op.reindex(close_s.index).ffill() - 1.0).rolling(20, min_periods=10).mean()

lib = {}
lib["mom20_volproxy60"] = per_asset(close_panel, mom20_volproxy60)
lib["dxy_beta_cond_60x20"] = per_asset(close_panel, beta_cond, macro_series("DXY"), 60, 20)
lib["calmness_20"] = per_asset(close_panel, calmness_20)
lib["intraday_drift_20"] = per_asset(close_panel, intraday_drift_20)

fwd = {}
for h in (1, 2, 3, 5, 10, 20):
    fwd[str(h)] = forward_returns(close_panel, h)

m = validate_factor(sig, close_panel, library=lib, fwd_cache=fwd)
print("usdjpy_beta_cond_120x60 validation:", json.dumps({k: (v if not isinstance(v, dict) else v)
                                                          for k, v in m.items() if k != "library_pairwise_corr"},
                                                         indent=1))
ic, icir = abs(m["ic"]), abs(m["icir"])
assert ic >= 0.007, "IC gate failed"
assert icir >= 0.084, "ICIR gate failed"
assert m["max_abs_library_correlation"] < 0.5, "library corr gate failed"

# ---------------------------------------------------------------------------
# 2. Build persistence JSON + signal artifact
# ---------------------------------------------------------------------------
regime_parts = []
ic_ser = compute_ic(sig, fwd["10"]).dropna()
for r0, r1 in [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
               ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-07-29"),
               ("2026-01-01", "2026-07-29")]:
    sub = ic_ser[(ic_ser.index >= r0) & (ic_ser.index <= r1)]
    if len(sub) >= 30:
        sd = sub.std()
        regime_parts.append(f"{r0[:4]}-{r1[:4]}: ic={sub.mean():+.4f} icir={sub.mean()/sd if sd>0 else 0:+.3f} n={len(sub)}")
    elif len(sub) > 0:
        regime_parts.append(f"{r0[:4]}-{r1[:4]}: n={len(sub)}")

doc = {
    "factor_id": "usdjpy_beta_cond_120x60",
    "factor_name": "USDJPY beta-conditional 120x60 (JPY carry-risk regime)",
    "version": "1.0.0",
    "calculation": {
        "expression": "rolling_beta(asset_daily_ret, USDJPY_daily_ret, 120d) * (USDJPY/USDJPY.shift(60)-1)",
        "description": "Per-asset 120d rolling beta of daily return on USDJPY return, multiplied by 60d USDJPY appreciation. Captures the classic carry-trade risk dimension: when JPY is depreciating (USDJPY up) over a 60d window, high-beta-to-JPY assets (risk assets) tend to outperform; when JPY is appreciating (safe-haven demand), positions flip defensively. Longer windows smooth the beta estimate and the 60d momentum term makes the factor regime-adaptive.",
    },
    "dependencies": ["close", "USDJPY"],
    "parameters": {"beta_window": 120, "momentum_window": 60, "min_periods_beta": 60},
    "expected_direction": 1,
    "validation": {
        "status": "EFFECTIVE",
        "period": "2020-01-01..2026-07-29",
        "last_validated": "2026-07-30",
        "admission_horizon": 10,
        "regime_notes": " | ".join(regime_parts),
        "metrics": {
            "ic": m["ic"],
            "icir": m["icir"],
            "ic_hit_ratio": m["ic_hit_ratio"],
            "n_ic_dates": m["n_ic_dates"],
            "coverage_asset_days": m["coverage_asset_days"],
            "coverage_dates_ge8": m["coverage_dates_ge8"],
            "n_dates_total": m["n_dates_total"],
            "n_dates_ge8": m["n_dates_ge8"],
            "turnover_10d_rank": m["turnover_10_rank"],
            "decay_ic_by_horizon": m["decay_ic_by_horizon"],
            "max_abs_library_correlation": m["max_abs_library_correlation"],
            "library_pairwise_corr": m["library_pairwise_corr"],
        },
    },
    "tags": ["macro-conditional", "JPY", "carry", "risk-on/off", "cross-asset"],
    "benchmark_admission": {
        "contract": {
            "ic_threshold": 0.007,
            "icir_threshold": 0.084,
            "correlation_threshold": 0.5,
            "library_capacity": 30,
            "active_top_k": 10,
        },
        "admission_gate_result": "PASS",
        "note": "IC 0.0456 / ICIR 0.1377 above thresholds; max abs library corr 0.0839 under 0.5 gate; turnover 0.14 compatible with 10d cadence.",
    },
}

# signal artifact
np.save("factors/usdjpy_beta_cond_120x60.signal.npy", sig.values)
doc["signal_artifact"] = "usdjpy_beta_cond_120x60.signal.npy"
doc["artifact_provenance"] = {
    "format": "npy_matrix",
    "shape": list(sig.shape),
    "columns": list(sig.columns),
    "dates_first": str(sig.index[0].date()),
    "dates_last": str(sig.index[-1].date()),
    "n_nan": int(np.isnan(sig.values).sum()),
}

with open("factors/usdjpy_beta_cond_120x60.json", "w") as fh:
    json.dump(doc, fh, indent=2)
print("WROTE factors/usdjpy_beta_cond_120x60.json")

# read-back verify
back = json.load(open("factors/usdjpy_beta_cond_120x60.json"))
sig2 = np.load("factors/usdjpy_beta_cond_120x60.signal.npy")
assert back["factor_id"] == "usdjpy_beta_cond_120x60"
assert back["validation"]["status"] == "EFFECTIVE"
assert back["validation"]["metrics"]["ic"] == m["ic"]
assert back["validation"]["metrics"]["icir"] == m["icir"]
assert back["signal_artifact"] == "usdjpy_beta_cond_120x60.signal.npy"
assert sig2.shape == sig.shape
assert np.allclose(sig2, sig.values, equal_nan=True)
print("READ-BACK OK")

# ---------------------------------------------------------------------------
# 3. Update ensemble (4-factor: mom20_volproxy60, dxy_beta_cond_60x20, calmness_20,
#    usdjpy_beta_cond_120x60) with pairwise rho gate on signal artifacts
# ---------------------------------------------------------------------------
from scipy.stats import rankdata

lib_ids = ["mom20_volproxy60", "dxy_beta_cond_60x20", "calmness_20", "usdjpy_beta_cond_120x60"]
meta = {}
for fid in lib_ids:
    p = f"factors/{fid}.json"
    d = json.load(open(p))
    assert d["validation"]["status"] in ("EFFECTIVE",), (fid, d["validation"]["status"])
    meta[fid] = {"artifact": f"factors/{fid}.signal.npy",
                 "last_validated": d["validation"]["last_validated"]}

# transformed signals (same transform as portfolio consumption):
def transform(sig):
    df = sig.copy()
    def _cs(x):
        r = pd.Series(x).rank(pct=True)
        z = (r - r.mean()) / (r.std() + 1e-9)
        return np.clip(z, -3, 3)
    return df.apply(_cs, axis=1)

arrs = {}
for fid in lib_ids:
    raw = pd.DataFrame(np.load(meta[fid]["artifact"]), index=close_panel.index,
                       columns=close_panel.columns)
    arrs[fid] = transform(raw.fillna(0.0))

print("\nPairwise Spearman rho (transformed signals):")
maxrho, pair = 0.0, None
K = len(lib_ids)
for i in range(K):
    for j in range(i + 1, K):
        a = arrs[lib_ids[i]].values.ravel()
        b = arrs[lib_ids[j]].values.ravel()
        msk = ~(np.isnan(a) | np.isnan(b))
        ra, rb = rankdata(a[msk]), rankdata(b[msk])
        rho = np.corrcoef(ra, rb)[0, 1]
        print(f"  rho {lib_ids[i]:28s} vs {lib_ids[j]:28s} = {rho:+.4f} (n={msk.sum()})")
        if abs(rho) > maxrho:
            maxrho, pair = abs(rho), (lib_ids[i], lib_ids[j], rho)
print(f"max pairwise |rho| = {maxrho:.4f} ({pair[0]} vs {pair[1]}) -> gate<0.5: {maxrho < 0.5}")
assert maxrho < 0.5, "transformed-signal correlation gate FAILED"

# quality-weighted weights
rows = []
for fid in lib_ids:
    d = json.load(open(f"factors/{fid}.json"))
    v = d["validation"]["metrics"]
    q = abs(v["ic"]) * abs(v["icir"])
    rows.append({"factor_id": fid, "ic": v["ic"], "icir": v["icir"],
                 "q": q, "last_validated": d["validation"]["last_validated"]})
rows.sort(key=lambda r: -r["q"])
total = sum(r["q"] for r in rows)
for r in rows:
    r["weight"] = r["q"] / total
    r["direction"] = 1 if r["ic"] >= 0 else -1

print("\nQuality-weighted rows:")
for r in rows:
    print(f"  {r['factor_id']:28s} ic={r['ic']:+.4f} icir={r['icir']:+.4f} q={r['q']:.6f} "
          f"w={r['weight']:.6f} dir={r['direction']}")
print("weights sum:", sum(r["weight"] for r in rows))

# build ensemble doc
ens = json.load(open("factors/factor_ensemble.json"))
cat_map = {
    "mom20_volproxy60": "Momentum (vol-damped)",
    "dxy_beta_cond_60x20": "Macro-Conditional (DXY beta)",
    "calmness_20": "Volatility (quiet-regime persistence)",
    "usdjpy_beta_cond_120x60": "Macro-Conditional (JPY carry-risk regime)",
}
selected = []
for r in rows:
    fid = r["factor_id"]
    selected.append({
        "factor_id": fid,
        "weight": round(r["weight"], 6),
        "direction": r["direction"],
        "ic": round(r["ic"], 4),
        "icir": round(r["icir"], 4),
        "quality": round(r["q"], 7),
        "signal_artifact": meta[fid]["artifact"],
        "admission_horizon": 10,
        "last_validated": meta[fid]["last_validated"],
        "transform": "cross-sectional rank, then z-score; winsorize 3 sigma",
        "category": cat_map[fid],
    })
selected.sort(key=lambda x: -x["quality"])
ens["selected_factors"] = selected
ens["weights_sum"] = round(sum(f["weight"] for f in selected), 6)
ens["updated_at"] = "2026-07-30"
ens["cycle"] = "2026-07-30"
ens["risk_notes"] = [
    "ACTIVE LIBRARY = 4 FACTORS: mom20_volproxy60, dxy_beta_cond_60x20, calmness_20, usdjpy_beta_cond_120x60 (all EFFECTIVE JSONs in persistence root).",
    "usdjpy_beta_cond_120x60 admitted: IC 0.0456 / ICIR 0.1377, q 0.006280, max lib corr 0.0839, turnover 0.14; all subperiods positive (2025-26 icir +0.303).",
    "eff_ratio_20 remains EVICTED (abs_spearman_rho 0.5240 vs mom20_volproxy60 > 0.5 gate); vol_surge_20 remains EVICTED (rho 0.5535 vs mom20_volproxy60).",
    "intraday_drift_20 remains QUARANTINED in factors/quarantine/ (reason: no recoverable signal artifact when first screened); root factors/intraday_drift_20.json re-validated but NOT ensemble-eligible until quarantine lifted.",
    "downside_dev_60 is DEPRECATED (sign flip in 2025-2026 regime breakdown; icir -0.025 on 2025-2026 subperiod).",
    "Pairwise transformed-signal rho gate re-run THIS cycle: max = {maxrho:.4f} ({pair[0]} vs {pair[1]}) < 0.5; all pairings under gate.".format(maxrho=maxrho, pair=pair),
    "Regime: VIX 23.76 (+44.5% in 20d) with DXY flat - liquidity-driven de-risking; JPY carry-risk factor adds regime-adaptive risk-off tilt orthogonal to vol-damped momentum and calmness.",
    "Signal coverage on final date (2026-07-29): 15/15 assets non-NaN for all four factors - ensemble fully usable.",
]
json.dump(ens, open("factors/factor_ensemble.json", "w"), indent=1, default=str)
print("\nWROTE factors/factor_ensemble.json | n_selected =", len(selected), "| weights_sum =", ens["weights_sum"])
print("DONE cycle31c persist usdjpy_beta_cond_120x60 + ensemble update")
