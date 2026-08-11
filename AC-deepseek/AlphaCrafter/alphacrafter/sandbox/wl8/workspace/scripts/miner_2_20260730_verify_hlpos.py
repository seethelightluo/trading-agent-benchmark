"""miner_2 2026-07-30 -- post-persistence verification + full-library correlation."""
import sys
import json
import base64
import zlib
import io
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, factor_panel,
                                   fwd_returns, ic_series, artifact_b64,
                                   IC_GATE, ICIR_GATE, ASSETS)

close, vol, open_, high, low = load_closes()
vix = load_index("VIX")
macro = {"VIX": vix, "DXY": load_index("DXY")}
macro["US10Y"] = close["US10Y"].dropna()
macro["CN10Y"] = close["CN10Y"].dropna()


def decode(fid):
    d = json.load(open(f"factors/{fid}.json"))
    raw = base64.b64decode(d["validation"]["signal_artifact"]["data"])
    p = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()), index_col=0, parse_dates=True)
    p.index = pd.DatetimeIndex(p.index)
    return d, p


# 1) verify persisted factors reload
for fid in ["hl_pos_150", "hl_pos_180"]:
    d, p = decode(fid)
    st = d["validation"]["status"]
    m = d["validation"]["metrics"]
    ok_ic = abs(m["ic"]) >= IC_GATE
    ok_icir = abs(m["icir"]) >= ICIR_GATE
    shape_ok = (d["validation"]["signal_artifact"]["shape"] == [int(p.shape[0]), int(p.shape[1])])
    print(f"[verify] {fid}: status={st} ic={m['ic']:.4f}(>={IC_GATE}:{ok_ic}) "
          f"icir={m['icir']:.4f}(>={ICIR_GATE}:{ok_icir}) artifact_shape={p.shape} ok={shape_ok} "
          f"max|rho|={m['max_abs_library_correlation']:.4f}", flush=True)

# 2) full-library correlation: recompute candidate panels, compare vs ALL 3 effective factors
def hl_pos(c, v, o, h, l, m, w, skip=0):
    hi = h.rolling(w).max().shift(skip)
    lo = l.rolling(w).min().shift(skip)
    rng = (hi - lo).replace(0, np.nan)
    return (c.shift(skip) - lo) / rng

lib = {}
for fid in ["mom_10d_skip5", "vix_beta_cond_60x20", "yield_beta_cond_60x20"]:
    _, lib[fid] = decode(fid)

for fid, w in [("hl_pos_150", 150), ("hl_pos_180", 180)]:
    panel = factor_panel(hl_pos, close, vol, open_, high, low, macro, w=w, skip=0)
    print(f"--- {fid} pooled pearson/spearman vs library ---", flush=True)
    for lfid, lp in lib.items():
        common = panel.index.intersection(lp.index)
        cols = [c for c in panel.columns if c in lp.columns]
        a = panel.loc[common, cols].values.ravel()
        b = lp.loc[common, cols].values.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        rho_p = float(np.corrcoef(a[m], b[m])[0, 1])
        from scipy.stats import spearmanr
        rho_s, _ = spearmanr(a[m], b[m])
        print(f"  vs {lfid}: pearson={rho_p:.4f} spearman={rho_s:.4f} n={m.sum()}", flush=True)
print("verify done", flush=True)
