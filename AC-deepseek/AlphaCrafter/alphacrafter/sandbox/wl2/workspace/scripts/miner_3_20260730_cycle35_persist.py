"""miner_3 cycle 35: persist cryptobeta_cond_60x20 (only cycle-35 gate passer).

IC=+0.0344 ICIR=+0.0872 hit=0.538, maxrho_raw=0.1967 (vs gain_loss_20),
maxrho_ranked=0.0985. Regime-stable positive except 2022 (crypto bear).
Note last250 ICIR=0.0673 below gate - documented as recency caveat.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (TRADABLES, load_panel, macro_series,
                         forward_returns, compute_ic, validate_factor)

FACTOR_ID = "cryptobeta_cond_60x20"
panel = load_panel()
HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}

# recompute the factor deterministically
btc = macro_series("BTC") if "BTC" not in panel.columns else panel["BTC"]
btc_ret = btc.pct_change()
btc_mom = btc / btc.shift(20) - 1.0
parts = {}
for a in TRADABLES:
    if a in ("BTC", "ETH"):
        parts[a] = pd.Series(np.nan, index=panel.index)
        continue
    s = panel[a].dropna()
    ar = s.pct_change()
    df = pd.concat([ar.rename("a"), btc_ret.rename("m")], axis=1).dropna()
    b = df["a"].rolling(60, min_periods=30).cov(df["m"]) / df["m"].rolling(60, min_periods=30).var()
    parts[a] = b.mul(btc_mom.reindex(b.index), axis=0).reindex(panel.index)
F = pd.DataFrame(parts, index=panel.index)

# active library (all top-level EFFECTIVE with .npy)
import glob
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
assert ic >= 0.007 and icir >= 0.084, f"{FACTOR_ID} fails IC gate: {m['ic']} {m['icir']}"
assert max_raw < 0.45 and max_ranked < 0.45, f"{FACTOR_ID} fails corr gate"
print("[metrics]", json.dumps(m, indent=1))
print("[regime]", reg_note)

doc = {
    "factor_id": FACTOR_ID,
    "factor_name": "Crypto-regime conditional beta 60x20",
    "version": "1.0.0",
    "calculation": {
        "expression": "beta(asset_ret, BTC_ret, 60d, min 30 obs) * (BTC/BTC.shift(20)-1); BTC/ETH columns NaN",
        "description": "Conditional macro-risk signal on the crypto axis: rolling beta of each asset's daily returns to BTC daily returns (60d) times the 20-day BTC move. Positive when an asset rises with crypto (high crypto-beta) while BTC trends up, or falls with crypto while BTC trends down. BTC/ETH excluded from their own beta (NaN) to avoid self-regression artifacts. In this 15-asset cross-market universe the raw rank IC is positive (IC +0.0344, ICIR +0.0872): high crypto-beta assets earn higher forward 10d returns when the crypto regime is strong. New axis vs the active library (max stacked rho 0.197 vs gain_loss_20).",
        "transform": "rank cross-sectionally (pct rank); portfolio uses direction=sign(IC)",
    },
    "dependencies": ["close", "BTC"],
    "parameters": {"beta_win": 60, "btc_win": 20, "min_periods": 30},
    "expected_direction": 1,
    "validation": {
        "status": "EFFECTIVE",
        "period": "2020-01-01..2026-07-29",
        "last_validated": "2026-07-30",
        "admission_horizon": 10,
        "regime_notes": f"15-instrument tradable cross-asset universe (13 assets with valid signal; BTC/ETH NaN). {reg_note}",
        "metrics": m,
    },
    "tags": ["macro", "crypto", "beta", "conditional", "cross-asset"],
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
        "admitted_at": "2026-08-11T02:10:00.000000",
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
          open("scripts/_miner3_cycle35_persist_results.json", "w"), indent=1, default=float)
print("\nDONE persist cycle35")
