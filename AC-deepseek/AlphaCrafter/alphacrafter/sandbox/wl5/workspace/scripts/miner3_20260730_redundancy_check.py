"""miner_3 2026-07-30: redundancy & provenance check for passing candidates."""
import sys
import json
import zlib
import base64
import io
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, macro_closes, forward_returns,
                             ic_series, load_panel)

VIS = "2026-07-29"
H = 10
close = closes_panel(VIS)
macro = macro_closes(VIS)
frames = load_panel(visible_through=VIS)
ret = close.pct_change()


def rolling_beta(asset_ret, mkt_ret, win, minp=40):
    out = {}
    for a in asset_ret.columns:
        pair = pd.concat([asset_ret[a].rename("a"), mkt_ret.rename("m")], axis=1).dropna()
        b = pair["a"].rolling(win, min_periods=minp).cov(pair["m"]) / pair["m"].rolling(win, min_periods=minp).var()
        out[a] = b
    return pd.DataFrame(out).reindex(asset_ret.index)


signals = {}
signals["NDX_BETA_60"] = rolling_beta(ret, ret["NDX"], 60)
signals["WTI_BETA_60"] = rolling_beta(ret, ret["WTI"], 60)
signals["ETH_BETA_60"] = rolling_beta(ret, ret["ETH"], 60)
mom20 = close / close.shift(20) - 1.0
signals["MOM_REL_EQ_20"] = mom20.sub(mom20.mean(axis=1), axis=0)

fr = forward_returns(close, H)
ics = {k: ic_series(v, fr, min_valid=8) for k, v in signals.items()}

print("=== pairwise IC-series correlation (candidate vs candidate) ===")
names = list(ics.keys())
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        pair = pd.concat([ics[names[i]].rename("a"), ics[names[j]].rename("b")], axis=1).dropna()
        r = pair["a"].corr(pair["b"])
        print(f"  {names[i]} vs {names[j]}: rho={r:.4f} (n={len(pair)})")

print("\n=== max_abs_library_correlation vs QUARANTINED library (provenance) ===")
qdir = "factors/quarantine"
lib_sigs = {}
for fn in sorted(os.listdir(qdir)):
    if not fn.endswith(".json") or fn.endswith(".reason.json"):
        continue
    d = json.load(open(f"{qdir}/{fn}"))
    art = d.get("validation", {}).get("signal_artifact")
    if not art:
        continue
    try:
        dec = zlib.decompress(base64.b64decode(art["data"])).decode("utf-8")
        s = pd.read_csv(io.StringIO(dec), index_col=0)
        s.index = pd.to_datetime(s.index)
        lib_sigs[d["factor_id"]] = s.reindex(close.index)
    except Exception as e:
        print("skip lib artifact", fn, e)
print("recoverable quarantined library signals:", list(lib_sigs.keys()))

for fid, s in signals.items():
    best = 0.0
    best_f = None
    for lf, ls in lib_sigs.items():
        lic = ic_series(ls, fr, min_valid=8)
        pair = pd.concat([ics[fid].rename("a"), lic.rename("b")], axis=1).dropna()
        if len(pair) >= 30:
            r = pair["a"].corr(pair["b"])
            if np.isfinite(r) and abs(float(r)) > best:
                best = abs(float(r))
                best_f = lf
    print(f"  {fid}: max_abs_lib_corr={best:.4f} (vs {best_f})")

print("\n=== signal-level (raw value) correlation among candidates ===")
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        pair = pd.concat([signals[names[i]].stack().rename("a"), signals[names[j]].stack().rename("b")], axis=1).dropna()
        r = pair["a"].corr(pair["b"])
        print(f"  {names[i]} vs {names[j]}: rho={r:.4f} (n={len(pair)})")
