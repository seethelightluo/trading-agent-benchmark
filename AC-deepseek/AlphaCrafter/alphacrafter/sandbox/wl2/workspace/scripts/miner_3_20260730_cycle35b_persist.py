"""miner_3 cycle 35b: persist volcluster_60 (vol-clustering persistence axis).

IC=+0.0362 ICIR=+0.1234 hit=0.554, maxrho_raw=0.1368, maxrho_ranked=0.1595 -
strongly orthogonal to the 12-factor active library. Positive IC in all four
regime sub-periods (weaker in 2023-26); last250 slightly negative (recency caveat).
"""
import sys, json, glob
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (TRADABLES, load_panel,
                         forward_returns, compute_ic, validate_factor)

FACTOR_ID = "volcluster_60"
panel = load_panel()
HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}


def per_asset_own(func):
    out = {}
    for a in TRADABLES:
        s = panel[a].dropna()
        out[a] = func(s).reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)


def volcluster(s, w=60, mp=40):
    r = s.pct_change().abs()
    return r.rolling(w, min_periods=mp).corr(r.shift(1))


F = per_asset_own(volcluster)

lib = {}
for npy in sorted(glob.glob("factors/*.signal.npy")):
    jp = npy.replace(".signal.npy", ".json")
    try:
        d = json.load(open(jp))
        if d.get("validation", {}).get("status") != "EFFECTIVE":
            continue
    except Exception:
        continue
    a = np.load(npy)
    if a.shape == (len(panel), len(panel.columns)):
        lib[Path(npy).stem.replace(".signal", "")] = pd.DataFrame(a, index=panel.index, columns=panel.columns)


def stacked_spearman(a, b, per_date_rank=False):
    aa = a.rank(axis=1) if per_date_rank else a
    bb = b.rank(axis=1) if per_date_rank else b
    df = pd.concat([aa.stack().rename("x"), bb.stack().rename("y")], axis=1).dropna()
    return float(df["x"].corr(df["y"], method="spearman")) if len(df) >= 30 else 0.0


m = validate_factor(F, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                    library=lib, fwd_cache=fwd_cache)
lc = {k: {"raw": round(stacked_spearman(F, sig), 4),
          "ranked": round(stacked_spearman(F, sig, per_date_rank=True), 4)}
      for k, sig in lib.items()}
max_raw = round(max((abs(v["raw"]) for v in lc.values()), default=0.0), 4)
max_ranked = round(max((abs(v["ranked"]) for v in lc.values()), default=0.0), 4)
m["max_abs_library_correlation"] = max_raw
m["max_ranked_library_correlation"] = max_ranked
m["library_pairwise_corr"] = lc
m["turnover_10d_rank"] = m.pop("turnover_10_rank", None)

ic_ser = compute_ic(F, fwd_cache[str(ADM_H)]).dropna()
reg = {}
for r0, r1, tag in [("2020-01-01", "2021-12-31", "2020-21"),
                    ("2022-01-01", "2022-12-31", "2022"),
                    ("2023-01-01", "2024-12-31", "2023-24"),
                    ("2025-01-01", "2026-07-29", "2025-26")]:
    sub = ic_ser[(ic_ser.index >= r0) & (ic_ser.index <= r1)]
    if len(sub) >= 30:
        sd = sub.std()
        reg[tag] = f"ic={sub.mean():+.4f} icir={(sub.mean()/sd if sd > 0 else 0):+.3f} n={len(sub)}"
last = ic_ser.iloc[-250:]
if len(last) >= 30:
    sd = last.std()
    reg["last250"] = f"ic={last.mean():+.4f} icir={(last.mean()/sd if sd > 0 else 0):+.3f} n={len(last)}"
reg_note = " | ".join(f"{k}:{v}" for k, v in reg.items())

ic, icir = abs(m["ic"]), abs(m["icir"])
assert ic >= 0.007 and icir >= 0.084, f"{FACTOR_ID} fails IC gate"
assert max_raw < 0.45 and max_ranked < 0.45, f"{FACTOR_ID} fails corr gate"
print("[metrics] ic=%s icir=%s hit=%s n=%s cov=%s maxrho_raw=%s maxrho_ranked=%s" % (
    m["ic"], m["icir"], m["ic_hit_ratio"], m["n_ic_dates"],
    m["coverage_asset_days"], max_raw, max_ranked))
print("[regime]", reg_note)

doc = {
    "factor_id": FACTOR_ID,
    "factor_name": "Volatility clustering persistence 60d",
    "version": "1.0.0",
    "calculation": {
        "expression": "rolling_corr(|daily_ret|, |daily_ret|.shift(1), 60d, min 40 obs) per asset own calendar",
        "description": "60-day autocorrelation of absolute daily returns: how persistently volatility clusters (GARCH-style persistence). High values identify assets whose vol shocks linger; low values identify assets whose vol regime is short-lived/mean-reverting. Raw rank IC positive (+0.0362, ICIR +0.1234): persistent-vol assets earn higher forward 10d cross-sectional returns in this universe - a vol-regime persistence premium distinct from calmness (quiet-day fraction) and vol-of-vol. Strongly orthogonal to the active library (max stacked rho 0.137 raw / 0.160 per-date-ranked).",
        "transform": "rank cross-sectionally (pct rank); portfolio uses direction=sign(IC)",
    },
    "dependencies": ["close"],
    "parameters": {"window": 60, "min_periods": 40, "lag": 1},
    "expected_direction": 1,
    "validation": {
        "status": "EFFECTIVE",
        "period": "2020-01-01..2026-07-29",
        "last_validated": "2026-07-30",
        "admission_horizon": 10,
        "regime_notes": f"15-instrument tradable cross-asset universe. {reg_note}",
        "metrics": m,
    },
    "tags": ["volatility", "clustering", "regime", "garch", "cross-asset"],
    "benchmark_admission": {
        "contract": {
            "ic_threshold": 0.007,
            "icir_threshold": 0.084,
            "correlation_threshold": 0.5,
            "library_capacity": 30,
            "active_top_k": 10,
        },
        "selected_metrics": {
            "ic": m["ic"],
            "icir": m["icir"],
            "metric_path": "validation.metrics",
            "reported_max_abs_library_correlation": m["max_abs_library_correlation"],
            "correlation_path": "validation.metrics.max_abs_library_correlation",
            "quality": round(abs(m["ic"]) * abs(m["icir"]), 8),
        },
        "admitted_at": "2026-08-11T02:40:00.000000",
    },
    "signal_artifact": f"{FACTOR_ID}.signal.npy",
    "artifact_provenance": {
        "format": "npy_matrix",
        "shape": list(F.shape),
        "columns": list(F.columns),
        "dates_first": str(F.index[0].date()),
        "dates_last": str(F.index[-1].date()),
        "n_nan": int((~F.notna()).sum().sum()),
    },
}
out = Path("factors") / f"{FACTOR_ID}.json"
out.write_text(json.dumps(doc, indent=1))
np.save(Path("factors") / f"{FACTOR_ID}.signal.npy", F.values)
print(f"[persist] wrote {out}")

chk = json.load(open(out))
ok = (chk["factor_id"] == FACTOR_ID
      and chk["validation"]["status"] == "EFFECTIVE"
      and abs(chk["validation"]["metrics"]["ic"]) >= 0.007
      and abs(chk["validation"]["metrics"]["icir"]) >= 0.084
      and Path("factors", chk["signal_artifact"]).exists()
      and np.load(Path("factors", chk["signal_artifact"])).shape == tuple(F.shape))
print(f"[verify] id={chk['factor_id'] == FACTOR_ID} status={chk['validation']['status']} "
      f"ic={chk['validation']['metrics']['ic']} icir={chk['validation']['metrics']['icir']} "
      f"rho={chk['validation']['metrics']['max_abs_library_correlation']} "
      f"artifact={Path('factors', chk['signal_artifact']).exists()} ALL_OK={ok}")

json.dump({"metrics": m, "regime": reg, "status": "EFFECTIVE"},
          open("scripts/_miner3_cycle35b_persist_results.json", "w"), indent=1, default=float)
print("\nDONE persist cycle35b")
