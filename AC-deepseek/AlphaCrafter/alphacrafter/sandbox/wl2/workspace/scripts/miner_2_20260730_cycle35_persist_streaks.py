"""miner_2 cycle35: persist max_consec_gain_20 & max_consec_loss_20.

Both passed IC/ICIR admission gates on 10d horizon:
  max_consec_gain_20 : IC=+0.0682 ICIR=+0.2310 hit=0.587 maxlibcorr=0.3418
  max_consec_loss_20 : IC=-0.0420 ICIR=-0.1432 hit=0.557 maxlibcorr=0.3253
Mutual rank rho = -0.219 (< 0.5), so both can coexist in the library.
pos_freq_20 (IC 0.045/ICIR 0.147) is NOT persisted: its rank rho vs
max_consec_gain_20 = +0.636 (> 0.5 gate) -> guaranteed quarantine; documented.

Saves signal artifacts .npy and writes JSON docs, then verifies read-back.
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "scripts")
from miner2_lib import (load_close_panel, per_asset, forward_returns,
                        compute_ic, validate_factor, library_correlation,
                        turnover_rank, coverage_stats, regime_breakdown)

panel = load_close_panel()
idx = panel.index


def max_consec(s, w=20, mp=10, direction=1):
    def _m(x):
        x = np.asarray(x, dtype=float)
        r = np.diff(x) / x[:-1]
        if len(r) < mp:
            return np.nan
        best = cur = 0
        for v in r:
            cur = cur + 1 if (v < 0 if direction < 0 else v > 0) else 0
            best = max(best, cur)
        return float(best)
    return s.rolling(w + 1, min_periods=mp + 1).apply(_m, raw=True)


EFF = ["mom20_volproxy60", "dxy_beta_cond_60x20", "calmness_20",
       "gain_loss_20", "intraday_drift_20", "usdjpy_beta_cond_120x60",
       "downside_dev_60", "days_since_high_60"]
lib = {}
for e in EFF:
    p = Path("factors") / f"{e}.signal.npy"
    if p.exists():
        a = np.load(p)
        if a.shape[0] == len(idx):
            lib[e] = pd.DataFrame(a, index=idx, columns=panel.columns)

fwd = {str(h): forward_returns(panel, h) for h in (1, 2, 3, 5, 10, 20)}

specs = {
    "max_consec_gain_20": {
        "direction": 1,
        "factor_name": "Longest Winning Streak (20d)",
        "expression": ("rolling 21d: longest run of consecutive up days in daily returns"),
        "description": ("Counts the longest consecutive run of positive daily returns within "
                        "the trailing 20 trading days. Persistence of daily gains (streak "
                        "momentum): assets currently stringing together winning sessions keep "
                        "outperforming over the next 10 days. Positive IC."),
        "interpretation": "high value = strong recent win streak; positive 10d IC",
    },
    "max_consec_loss_20": {
        "direction": -1,
        "factor_name": "Longest Losing Streak (20d)",
        "expression": ("rolling 21d: longest run of consecutive down days in daily returns"),
        "description": ("Counts the longest consecutive run of negative daily returns within "
                        "the trailing 20 trading days. Persistence of daily losses: assets with "
                        "long losing streaks keep lagging over the next 10 days (weak hands "
                        "continue to exit / no dip-buying yet). Negative IC."),
        "interpretation": "high value = recent long losing streak; negative 10d IC",
    },
}

for fid, sp in specs.items():
    f = per_asset(panel, max_consec, 20, 10, sp["direction"])
    art_path = Path("factors") / f"{fid}.signal.npy"
    np.save(art_path, f.values)

    lc = library_correlation(f, lib)
    m = validate_factor(f, panel, library=lib, fwd_cache=fwd)
    ic_ser = compute_ic(f, fwd["10"]).dropna()
    ic = float(ic_ser.mean())
    icir = float(ic_ser.mean() / ic_ser.std())
    to = turnover_rank(f, step=10)
    cov = coverage_stats(f)
    reg = regime_breakdown(ic_ser)
    print(f"[{fid}] IC={ic:+.4f} ICIR={icir:+.4f} turnover={to if to == to else None} "
          f"maxlibcorr={lc['max_abs']:.4f}")

    metrics = {
        "ic": round(ic, 4),
        "icir": round(icir, 4),
        "ic_hit_ratio": round(float((np.sign(ic_ser) == np.sign(ic)).mean()), 3),
        "n_ic_dates": int(len(ic_ser)),
        "coverage_asset_days": cov["coverage_asset_days"],
        "coverage_dates_ge8": cov["coverage_dates_ge8"],
        "n_dates_total": cov["n_dates_total"],
        "n_dates_ge8": cov["n_dates_ge8"],
        "turnover_10d_rank": round(to, 4) if to == to else None,
        "max_abs_library_correlation": round(lc["max_abs"], 4),
        "library_pairwise_corr": {k: round(v, 4) for k, v in lc["pairwise"].items()},
        "decay_ic_by_horizon": m["decay_ic_by_horizon"],
        "signal_artifact": str(art_path),
    }

    doc = {
        "factor_id": fid,
        "factor_name": sp["factor_name"],
        "version": "1.0.0",
        "calculation": {"expression": sp["expression"],
                        "description": sp["description"],
                        "interpretation": sp["interpretation"]},
        "dependencies": ["close"],
        "parameters": {"window": 20, "min_periods": 10, "admission_horizon": 10},
        "validation": {
            "status": "EFFECTIVE",
            "validated_at": "2026-07-30",
            "period": "2020-01-01..2026-07-29",
            "admission_gate": {"ic_abs_min": 0.0070, "icir_abs_min": 0.0840},
            "metrics": metrics,
            "regime_notes": {k: v for k, v in reg.items()},
            "universe": "15-instrument tradable cross-asset benchmark",
        },
        "tags": ["return-consistency", "streak", "behavioral", "momentum-like"],
        "last_validated": "2026-07-30",
    }
    out = Path("factors") / f"{fid}.json"
    json.dump(doc, open(out, "w"), indent=1, default=str)
    print("wrote", out)

    chk = json.load(open(out))
    assert chk["factor_id"] == fid
    assert chk["validation"]["status"] == "EFFECTIVE"
    assert Path(chk["validation"]["metrics"]["signal_artifact"]).exists()
    assert abs(chk["validation"]["metrics"]["ic"]) >= 0.007
    assert abs(chk["validation"]["metrics"]["icir"]) >= 0.084
    print("VERIFIED:", chk["factor_id"], chk["validation"]["status"],
          "| ic", chk["validation"]["metrics"]["ic"],
          "| icir", chk["validation"]["metrics"]["icir"],
          "| maxlibcorr", chk["validation"]["metrics"]["max_abs_library_correlation"])
    print()
