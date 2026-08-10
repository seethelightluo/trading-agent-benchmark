"""miner_2: Explore macro-beta factor family (VIX/USDJPY/USDCNY/EURUSD/DXY-conditional).

Compute rolling 60d betas of each asset's returns to each macro signal, IC/ICIR
at h=10, regime splits, and pairwise IC-series correlation vs persisted
dxy_beta_60 (loaded from its signal artifact) to manage library correlation.
"""
import sys, json, base64, zlib, io
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_validate import (closes_panel, macro_closes, forward_returns,
                             ic_series, summary_metrics, regime_split, WATCH)

VIS = "2026-07-29"
H = 10
WIN = 60

close = closes_panel(VIS)
macro = macro_closes(VIS)
ret = close.pct_change()
fr = forward_returns(close, H)


def rolling_beta(asset_ret, mkt_ret, win):
    out = {}
    for a in asset_ret.columns:
        pair = pd.concat([asset_ret[a].rename("a"), mkt_ret.rename("m")], axis=1).dropna()
        b = pair["a"].rolling(win).cov(pair["m"]) / pair["m"].rolling(win).var()
        out[a] = b
    return pd.DataFrame(out).reindex(asset_ret.index)


def load_lib_ic(fid):
    with open(f"factors/{fid}.json") as f:
        meta = json.load(f)
    a = meta["validation"]["signal_artifact"]
    dec = zlib.decompress(base64.b64decode(a["data"])).decode("utf-8")
    sig = pd.read_csv(io.StringIO(dec), index_col=0)
    sig.index = pd.to_datetime(sig.index)
    return ic_series(sig.reindex(close.index), fr, min_valid=8)


lib_ic = {"dxy_beta_60": load_lib_ic("dxy_beta_60")}

cands = {
    "vix_beta_60": lambda: rolling_beta(ret, macro["VIX"].pct_change(), WIN),
    "usdjpy_beta_60": lambda: rolling_beta(ret, macro["USDJPY"].pct_change(), WIN),
    "usdcny_beta_60": lambda: rolling_beta(ret, macro["USDCNY"].pct_change(), WIN),
    "eurusd_beta_60": lambda: rolling_beta(ret, macro["EURUSD"].pct_change(), WIN),
    "dxy_beta_cond_60x20": lambda: -rolling_beta(ret, macro["DXY"].pct_change(), WIN)
        * (macro["DXY"] / macro["DXY"].shift(20) - 1.0),
    "vix_beta_cond_60x20": lambda: -rolling_beta(ret, macro["VIX"].pct_change(), WIN)
        * (macro["VIX"] / macro["VIX"].shift(20) - 1.0),
}

results = {}
for name, fn in cands.items():
    f = fn().reindex(close.index)
    ic = ic_series(f, fr, min_valid=8)
    m = summary_metrics(ic, f, fr, close, h=H)
    m["regime"] = regime_split(ic)
    rho = {}
    for lid, lic in lib_ic.items():
        pair = pd.concat([ic.rename("a"), lic.rename("b")], axis=1).dropna()
        rho[lid] = round(float(pair["a"].corr(pair["b"])), 4) if len(pair) > 30 else None
    m["rho_vs_library"] = rho
    results[name] = {"ic": ic, "factor": f, "metrics": m}
    print(f"\n=== {name} ===")
    for k, v in m.items():
        if k in ("regime", "rho_vs_library"):
            print(f" {k}: {json.dumps(v)}")
        elif k == "decay_ic_by_horizon":
            print(f" decay: {v}")
        else:
            print(f" {k}: {v}")

print("\n--- gate check ---")
for name, r in results.items():
    mm = r["metrics"]
    g1 = abs(mm["ic"]) >= 0.007
    g2 = abs(mm["icir"] or 0) >= 0.084
    rho_max = max([abs(v) for v in mm["rho_vs_library"].values() if v is not None] or [0])
    print(f"{name}: IC={mm['ic']:.4f} ICIR={mm['icir']} n={mm['n_ic_dates']} "
          f"cov={mm['coverage_dates_ge8']:.2f} rho_max={rho_max:.3f} "
          f"PASS={g1 and g2 and rho_max < 0.5}")
