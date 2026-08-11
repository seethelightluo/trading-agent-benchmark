"""Screener: build factor ensemble for cycle 2026-07-30.

Method: quality_ic_tilt
  q_raw      = |IC| * |ICIR|                          (persisted validation.metrics)
  direction  = sign(IC)                               (preserve persisted sign)
  turnover_penalty = clip(1 - 0.12*max(0, turnover_10d_rank - 1.5), 0.4, 1.0)
  corr_bonus = 1 + 0.25 * max(0, 0.35 - max_abs_library_correlation)
  w          = q_raw * turnover_penalty * corr_bonus, normalized to sum 1
"""
import json, math
from pathlib import Path

FACTOR_DIR = Path("factors")
OUT = Path("factor_ensemble.json")

FACTORS = [
    "amihud_20", "beta_ew_60d", "downside_vol_ratio_20", "max_ret_20d",
    "mom_10d_skip5", "mom_120d_skip5", "rel_mom_20d_skip5",
    "vix_beta_cond_60x20", "vol_of_vol20x60",
]

rows = []
for fid in FACTORS:
    d = json.loads((FACTOR_DIR / f"{fid}.json").read_text())
    m = d["validation"]["metrics"]
    ic, icir = m["ic"], m["icir"]
    q = abs(ic) * abs(icir)
    direction = 1 if ic >= 0 else -1
    turnover = m.get("turnover_10d_rank", 1.0)
    maxcorr = m.get("max_abs_library_correlation", 0.0)
    t_pen = max(0.4, 1.0 - 0.12 * max(0.0, turnover - 1.5))
    c_bonus = 1.0 + 0.25 * max(0.0, 0.35 - maxcorr)
    adj = q * t_pen * c_bonus
    rows.append({
        "factor_id": fid, "ic": ic, "icir": icir, "q": q,
        "direction": direction, "turnover": turnover, "maxcorr": maxcorr,
        "t_pen": t_pen, "c_bonus": c_bonus, "adj": adj,
        "category": d.get("tags", [])[:1],
    })

total = sum(r["adj"] for r in rows)
for r in rows:
    r["weight"] = r["adj"] / total

rows.sort(key=lambda r: -r["weight"])
print(f"{'factor':<24}{'dir':>4}{'q':>9}{'turn':>7}{'mxcorr':>8}{'t_pen':>7}{'c_bon':>7}{'weight':>9}")
for r in rows:
    print(f"{r['factor_id']:<24}{r['direction']:>4}{r['q']:>9.5f}{r['turnover']:>7.2f}"
          f"{r['maxcorr']:>8.3f}{r['t_pen']:>7.3f}{r['c_bonus']:>7.3f}{r['weight']:>9.4f}")
print(f"\nsum weights = {sum(r['weight'] for r in rows):.6f}  (n={len(rows)})")
mom_w = sum(r["weight"] for r in rows if r["factor_id"] in ("mom_10d_skip5", "mom_120d_skip5", "rel_mom_20d_skip5"))
risk_w = sum(r["weight"] for r in rows if r["factor_id"] in ("downside_vol_ratio_20", "max_ret_20d", "vol_of_vol20x60", "beta_ew_60d", "vix_beta_cond_60x20"))
print(f"momentum-cluster weight = {mom_w:.4f}, risk/defensive-cluster weight = {risk_w:.4f}")

ensemble = {
    "schema_version": 1,
    "selected_factors": [
        {"factor_id": r["factor_id"], "weight": round(r["weight"], 4), "direction": r["direction"]}
        for r in rows
    ],
    "method": "quality_ic_tilt",
}
OUT.write_text(json.dumps(ensemble, indent=2))
print(f"\nwrote {OUT} with {len(rows)} factors")
