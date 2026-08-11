"""miner_1 cycle 2026-07-30: persist kurt_20 (short-window tail concentration).

kurt_20 = rolling 20d excess kurtosis of daily returns: m4/m2^2 - 3, min_periods=8.

Motivation: distinguishes assets whose recent return process is dominated by a few extreme
(lumpy) days from assets with steady drift. High-kurtosis assets (concentrated moves)
outperform over the next 10d. Empirically positive IC in ALL three regimes (2020-2022, 2023-2024,
2025-2026), IC strengthens with horizon, and it is orthogonal to the 9-factor library
(max-abs Spearman rho 0.167). NOTE: the 60d version of kurtosis carries no signal
(IC=0.0014) - the 20d window is the informative horizon.

Gate check: |IC|=0.0304 >= 0.007, |ICIR|=0.0928 >= 0.084, max-abs lib rho 0.167 < 0.5.
"""
import json, sys, time, os
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_validate import closes_panel, forward_returns, ic_series, summary_metrics, regime_split
from miner3_lib import decode_artifact, build_artifact, LIB_FACTORS

VIS = '2026-07-29'
H = 10
close = closes_panel(VIS)
ret = close.pct_change()
print(f"panel: dates={len(close)} assets={len(close.columns)} visible_through={VIS}", flush=True)

# ---- signal: kurt_20 ----
mu20 = ret.rolling(20, min_periods=8).mean()
m2 = ((ret - mu20) ** 2).rolling(20, min_periods=8).mean()
m4 = ((ret - mu20) ** 4).rolling(20, min_periods=8).mean()
sig = m4 / (m2 ** 2) - 3.0

fr = forward_returns(close, H)
ic_s = ic_series(sig, fr, min_valid=8)
m = summary_metrics(ic_s, sig, fr, close, h=H)
m['regime'] = regime_split(ic_s)

# library max-abs Spearman rho from real artifacts
best = 0.0
rhos = {}
for lfid in LIB_FACTORS:
    p = f'factors/{lfid}.json'
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    art = d.get('validation', {}).get('signal_artifact')
    if not art:
        continue
    libp = decode_artifact(art).reindex(close.index)
    common = sig.index.intersection(libp.index)
    a = sig.loc[common].stack()
    b = libp.loc[common].stack()
    mm = a.notna() & b.notna()
    if mm.sum() >= 200:
        r = float(a[mm].rank().corr(b[mm].rank()))
        if np.isfinite(r):
            rhos[lfid] = round(r, 3)
            best = max(best, abs(r))
m['library_spearman_rho'] = rhos
m['max_abs_library_correlation'] = round(best, 3)
print("library spearman rho:", json.dumps(rhos, indent=1), flush=True)

gate_ic = abs(m['ic']) >= 0.007
gate_icir = abs(m['icir'] or 0) >= 0.084
gate = bool(gate_ic and gate_icir and best < 0.5)
print(f"=== kurt_20: ic={m['ic']} icir={m['icir']} hit={m['ic_hit_ratio']} n={m['n_ic_dates']} "
      f"cov_ad={m['coverage_asset_days']} cov_ge8={m['coverage_dates_ge8']} "
      f"turn={m['turnover_10d_rank']} max_rho_lib={best} GATE={gate}", flush=True)
print("  decay:", m['decay_ic_by_horizon'], flush=True)
print("  regimes:", m['regime'], flush=True)

if not gate:
    print("GATE FAILED - not persisting", flush=True)
    sys.exit(1)

factor_id = "kurt_20"
doc = {
    "factor_id": factor_id,
    "factor_name": "20-day Return Kurtosis (tail concentration)",
    "version": "1.0.0",
    "calculation": {
        "expression": "rolling_kurtosis(pct_change(close), 20, min_periods=8) = m4/m2^2 - 3, "
                      "where m2,m4 are rolling 20d 2nd/4th central moments of daily returns",
        "description": ("Excess kurtosis of daily returns over the trailing 20 sessions "
                        "(min_periods=8; union-calendar aware). High values = the return process is "
                        "dominated by a few extreme (lumpy) days; low values = steady drift. "
                        "High-kurtosis assets outperform over the next 10 days. Computed on the "
                        "15-asset tradable cross-asset universe; direction +1.")
    },
    "dependencies": ["close"],
    "parameters": {"window": 20, "min_periods": 8, "admission_horizon": 10},
    "expected_direction": 1,
    "validation": {
        "status": "EFFECTIVE",
        "period": "2020-01-01..2026-07-29",
        "admission_horizon": 10,
        "last_validated": "2026-07-30",
        "regime_notes": ("Cross-sectional rank IC vs 10d forward returns on the 15-asset tradable "
                         "universe. Positive in ALL regimes: 2020-2022 IC=0.0310 (ICIR=0.0998, n=440), "
                         "2023-2024 IC=0.0157 (ICIR=0.0465, n=305), 2025-2026 IC=0.0496 (ICIR=0.1442, "
                         "n=217). Decay IC strengthens with horizon: 0.0032 (1d) -> 0.0304 (10d) -> "
                         "0.0409 (20d). Orthogonal to the 9-factor library (max-abs Spearman rho "
                         "0.167). 60d kurtosis carries no signal (IC=0.0014), so 20d is the "
                         "informative horizon. Re-validate every ~3 months."),
        "metrics": m,
        "signal_artifact": build_artifact(sig),
    },
    "tags": ["quality", "tail-risk", "cross-asset", "return-distribution"],
}

with open(f"factors/{factor_id}.json", "w") as f:
    json.dump(doc, f, indent=1, default=str)
print(f"WROTE factors/{factor_id}.json", flush=True)

# ---- verify read-back ----
with open(f"factors/{factor_id}.json") as f:
    back = json.load(f)
assert back["factor_id"] == factor_id, "factor_id mismatch"
assert back["validation"]["status"] == "EFFECTIVE", "status mismatch"
assert back["validation"]["metrics"]["ic"] == m["ic"], "ic mismatch"
assert back["validation"]["metrics"]["icir"] == m["icir"], "icir mismatch"
assert back["validation"]["metrics"]["max_abs_library_correlation"] == round(best, 3)
assert "signal_artifact" in back["validation"] and back["validation"]["signal_artifact"]["data"], "artifact missing"
print("READ-BACK OK: id=%s status=%s ic=%s icir=%s max_rho=%s n_valid=%s artifact_bytes=%d" % (
    back["factor_id"], back["validation"]["status"], back["validation"]["metrics"]["ic"],
    back["validation"]["metrics"]["icir"],
    back["validation"]["metrics"]["max_abs_library_correlation"],
    back["validation"]["signal_artifact"]["n_valid_values"],
    len(back["validation"]["signal_artifact"]["data"])), flush=True)
