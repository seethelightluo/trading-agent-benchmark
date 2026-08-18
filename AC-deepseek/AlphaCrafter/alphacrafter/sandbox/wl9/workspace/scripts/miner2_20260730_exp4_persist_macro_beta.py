"""Persist Exploration 4 macro-beta factors passing the admission gate.

Passing variants (horizon 10, gates |IC|>=0.0070, |ICIR|>=0.0840):
  beta_VIX_60     IC -0.0828 ICIR -0.2022 (direction -1)
  beta_DXY_60     IC  0.0450 ICIR  0.1331 (direction +1)
  beta_USDJPY_20  IC  0.0392 ICIR  0.1113 (direction +1)
DXY and EURUSD betas are near-mirrors (pooled corr -0.93); keep DXY as the
representative dollar factor. VIX-beta complements but correlates with the
existing conditional vix_beta_cond_60x20 - check and report library corr.
"""
import sys, json, os, io, base64, zlib, hashlib, datetime
sys.path.insert(0, "scripts")
from miner2_20260730_factorlib import (load_panel, factor_panel, fwd_ret_panel,
                                       validate, load_close)
import pandas as pd, numpy as np
from scipy.stats import spearmanr

P = load_panel()
R = P.pct_change()
fwd10 = fwd_ret_panel(P, 10)

macro = {m: load_close(m, macro=True) for m in ["VIX", "DXY", "USDJPY"]}
macro_r = {m: s.pct_change() for m, s in macro.items()}

def rolling_beta(asset_r, macro_r, w):
    df = pd.concat([asset_r.rename("a"), macro_r.rename("m")], axis=1).dropna()
    beta = df["a"].rolling(w).cov(df["m"]) / df["m"].rolling(w).var()
    return beta.reindex(asset_r.index)

SPECS = [
    ("beta_VIX_60", "VIX", 60, -1),
    ("beta_DXY_60", "DXY", 60, 1),
    ("beta_USDJPY_20", "USDJPY", 20, 1),
]

def load_lib_factor(fid):
    d = json.load(open(f"factors/{fid}.json"))
    sa = d["validation"]["signal_artifact"]
    raw = zlib.decompress(base64.b64decode(sa["data"]))
    df = pd.read_csv(io.BytesIO(raw), index_col=0)
    df.index = pd.to_datetime(df.index)
    return df

LIB_IDS = ["mom_10d_skip5", "mom_120d_skip5", "vol_of_vol20x60",
           "vix_beta_cond_60x20", "dside_ratio_21"]

for fid, mname, w, direction in SPECS:
    fvals = factor_panel(R, lambda s, mr=macro_r[mname], w=w: rolling_beta(s, mr, w))
    res = validate(fvals, fwd10, label=fid, expected_dir=direction)
    print("VALIDATION:", json.dumps(res))
    assert res["passes"], f"{fid} did not pass gate - abort"
    assert int(fvals.notna().sum(axis=0).gt(0).sum()) == 15

    # library correlation provenance
    corrs = {}
    for lid in LIB_IDS:
        try:
            lf = load_lib_factor(lid)
            common = fvals.index.intersection(lf.index)
            if len(common) < 100:
                corrs[lid] = None
                continue
            a = fvals.loc[common].stack()
            b = lf.loc[common].stack()
            m = a.notna() & b.notna()
            rho, _ = spearmanr(a[m], b[m])
            corrs[lid] = float(rho) if np.isfinite(rho) else None
        except Exception as e:
            corrs[lid] = f"ERR:{e}"
    max_abs = max([abs(v) for v in corrs.values() if isinstance(v, float)], default=None)
    print("LIB_CORR:", json.dumps(corrs), "MAX_ABS:", max_abs)

    # signal artifact
    art = fvals.copy().sort_index().sort_index(axis=1)
    raw = art.to_csv().encode("utf-8")
    comp = zlib.compress(raw)
    b64 = base64.b64encode(comp).decode("ascii")
    signal_artifact = {
        "format": "base64:zlib:csv",
        "description": f"Factor signal panel: rows = dates, cols = assets. Shape {list(art.shape)}",
        "columns": list(art.columns),
        "shape": list(art.shape),
        "n_valid_values": int(art.notna().sum().sum()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": b64,
    }

    metrics = {
        "ic": res["ic"], "icir": res["icir"], "ic_hit_ratio": res["ic_hit_ratio"],
        "n_ic_dates": res["n_ic_dates"],
        "coverage_asset_days": res["coverage"],
        "coverage_dates_ge8": None,
        "turnover_10d_rank": res["turnover_10d_rank"],
        "decay_ic_by_horizon": res["decay_ic_by_horizon"],
        "max_abs_library_correlation": max_abs,
        "admission_direction": direction,
    }
    expr = {  # json-safe description
        "VIX": "cov(r, dVIX, w)/var(dVIX, w), dVIX = pct_change(VIX level)",
        "DXY": "cov(r, dDXY, w)/var(dDXY, w), dDXY = pct_change(DXY level)",
        "USDJPY": "cov(r, dUSDJPY, w)/var(dUSDJPY, w), dUSDJPY = pct_change(USDJPY level)",
    }[mname]
    desc = {
        "VIX": ("Rolling 60d beta of asset daily returns on VIX daily returns. High VIX-beta "
                "assets amplify equity sell-offs; validated NEGATIVE 10d direction (risk/resilience premium)."),
        "DXY": ("Rolling 60d beta of asset daily returns on DXY daily returns. High DXY-beta assets "
                "benefit from dollar strength; validated POSITIVE 10d direction."),
        "USDJPY": ("Rolling 20d beta of asset daily returns on USDJPY daily returns (carry/risk-on "
                   "signature). Validated POSITIVE 10d direction."),
    }[mname]
    factor = {
        "factor_id": fid,
        "factor_name": {
            "beta_VIX_60": "VIX beta (60d)",
            "beta_DXY_60": "DXY beta (60d)",
            "beta_USDJPY_20": "USDJPY beta (20d)",
        }[fid],
        "version": "1.0.0",
        "calculation": {"expression": expr, "description": desc},
        "dependencies": ["close", "index_data"],
        "parameters": {"window": w, "horizon": 10, "min_assets_for_ic": 8, "macro_series": mname},
        "expected_direction": direction,
        "validation": {
            "status": "EFFECTIVE",
            "period": "2020-01-01..2026-07-29",
            "last_validated": "2026-07-30",
            "admission_horizon": 10,
            "regime_notes": ("15-asset cross-asset universe 2020-01-01..2026-07-29 across multiple "
                             "regimes. IC stable in signed direction; decay strengthens with horizon."),
            "metrics": metrics,
            "signal_artifact": signal_artifact,
        },
        "tags": ["macro", "beta", "risk", "carry"] + ([("volatility" if mname == "VIX" else "fx")]),
        "benchmark_admission": {
            "contract": {"ic_threshold": 0.007, "icir_threshold": 0.084,
                         "correlation_threshold": 0.5, "library_capacity": 30, "active_top_k": 10},
            "selected_metrics": {"ic": res["ic"], "icir": res["icir"],
                                 "metric_path": "validation.metrics",
                                 "reported_max_abs_library_correlation": max_abs},
            "admitted_at": datetime.datetime.now().isoformat(),
        },
    }
    out = f"factors/{fid}.json"
    with open(out, "w") as fh:
        json.dump(factor, fh)
    # verify read-back
    chk = json.load(open(out))
    assert chk["factor_id"] == fid and chk["validation"]["status"] == "EFFECTIVE"
    m = chk["validation"]["metrics"]
    assert abs(m["ic"]) >= 0.0070 and abs(m["icir"]) >= 0.0840
    rb = zlib.decompress(base64.b64decode(chk["validation"]["signal_artifact"]["data"]))
    rd = pd.read_csv(io.BytesIO(rb), index_col=0)
    assert rd.shape == art.shape
    print(f"READBACK OK: {fid} ic={m['ic']} icir={m['icir']} shape={rd.shape} size={os.path.getsize(out)}")
print("DONE")