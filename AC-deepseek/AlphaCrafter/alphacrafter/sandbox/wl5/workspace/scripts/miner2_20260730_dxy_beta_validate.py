"""miner_2: Validate DXY-beta factor (rolling beta of asset returns to DXY returns).

Idea: assets with persistent sensitivity to the US dollar index (DXY) may be
differentially exposed to global liquidity / dollar funding regimes. We test
whether cross-sectional differences in rolling DXY-beta predict forward 10d
returns on the 15-asset tradable universe.

Validates window variants 30/60/90, both signs, decay, regime splits, and
library correlation. Only the 60d variant is the primary candidate.
"""
import sys, json, base64, zlib, io
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_validate import (closes_panel, macro_closes, forward_returns,
                             ic_series, summary_metrics, regime_split,
                             library_ic_series_map, max_abs_library_corr,
                             WATCH)

VIS = "2026-07-29"
H = 10

close = closes_panel(VIS)
macro = macro_closes(VIS)
ret = close.pct_change()
print("panel:", close.shape, "dates:", close.index.min().date(), "..", close.index.max().date())
print("instruments:", len(close.columns))

dxy = macro["DXY"]
dxy_ret = dxy.pct_change()


def dxy_beta(win):
    out = {}
    for a in close.columns:
        pair = pd.concat([ret[a].rename("a"), dxy_ret.rename("d")], axis=1).dropna()
        b = pair["a"].rolling(win).cov(pair["d"]) / pair["d"].rolling(win).var()
        out[a] = b
    return pd.DataFrame(out).reindex(close.index)


fr = forward_returns(close, H)
results = {}
for win in (30, 60, 90):
    f = dxy_beta(win)
    ic = ic_series(f, fr, min_valid=8)
    m = summary_metrics(ic, f, fr, close, h=H)
    m["n_assets"] = int(f.notna().any(axis=0).sum())
    # library correlation
    lib = library_ic_series_map(close, h=H)
    m["max_abs_library_correlation"] = max_abs_library_corr(ic, lib)
    m["regime"] = regime_split(ic)
    results[win] = {"ic_series": ic, "factor": f, "metrics": m}

for win in (30, 60, 90):
    m = results[win]["metrics"]
    print(f"\n=== DXY beta win={win} ===")
    for k, v in m.items():
        if k == "regime":
            print(" regime:", json.dumps(v))
        elif k == "decay_ic_by_horizon":
            print(" decay:", v)
        else:
            print(f" {k}: {v}")

# Best-direction check: does negative beta (dollar hedge) earn premium?
for win in (30, 60, 90):
    ic = results[win]["ic_series"]
    f = results[win]["factor"]
    m = results[win]["metrics"]
    gate_ic = abs(m["ic"]) >= 0.007
    gate_icir = abs(m["icir"] or 0) >= 0.084
    print(f"\nwin={win}: |IC|={abs(m['ic']):.4f} gate={gate_ic}, "
          f"|ICIR|={abs(m['icir'] or 0):.4f} gate={gate_icir}, "
          f"n_ic_dates={m['n_ic_dates']}, PASS={gate_ic and gate_icir}")
