"""miner_3 2026-07-30: full gate validation + redundancy/provenance check
for candidates NDX_BETA_60, WTI_BETA_60, ETH_BETA_60, MOM_REL_EQ_20."""
import sys, json, zlib, base64, io
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, macro_closes, forward_returns,
                             ic_series, summary_metrics, regime_split,
                             library_ic_series_map, max_abs_library_corr, WATCH)

VIS = "2026-07-29"
H = 10
close = closes_panel(VIS)
macro = macro_closes(VIS)
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

print("=== GATE METRICS (h=10, min_valid=8) ===")
for fid, s in signals.items():
    m = summary_metrics(ics[fid], s, fr, close, h=H)
    m["regime"] = regime_split(ics[fid])
    m["n_instruments"] = int(s.notna().sum().sum() / max(1, len(s)))
    print(json.dumps({"factor": fid, **{k: v for k, v in m.items() if k != "decay_ic_by_horizon"}}, default=str))
    print("   decay:", json.dumps(m["decay_ic_by_horizon"]))
    gate_ic = abs(m["ic"]) >= 0.007
    gate_icir = abs(m["icir"] or 0) >= 0.084
    print(f"   -> IC gate: {gate_ic} (|{m['ic']}|>=0.007), ICIR gate: {gate_icir} (|{m['icir']}|>=0.084)")

print("\n=== pairwise IC-series correlation (candidate vs candidate) ===")
names = list(ics.keys())
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        pair = pd.concat([ics[names[i]].rename("a"), ics[names[j]].rename("b")], axis=1).dropna()
        r = pair["a"].corr(pair["b"])
        print(f"  {names[i]} vs {names[j]}: rho={r:.4f} (n={len(pair)})")

print("\n=== max_abs_library_correlation vs EFFECTIVE library (factors/) ===")
lib = library_ic_series_map(close, h=H)
print("effective library entries:", list(lib.keys()))
for fid in names:
    print(f"  {fid}: {max_abs_library_corr(ics[fid], lib):.4f}")

print("\n=== max_abs_library_correlation vs QUARANTINED factors (provenance) ===")
qdir = "factors/quarantine"
lib_sigs = {}
for fn in sorted(os.listdir(qdir)):
    if not fn.endswith(".json") or fn.endswith(".reason.json"):
        continue
    d = json.load(open(f"{qdir}/{fn}"))
    art = d.get("validation", {}).get("signal_artifact")
    if not art or "data" not in art:
        continue
    try:
        dec = zlib.decompress(base64.b64decode(art["data"])).decode("utf-8")
        s = pd.read_csv(io.StringIO(dec), index_col=0)
        s.index = pd.to_datetime(s.index)
        lib_sigs[d["factor_id"]] = s.reindex(close.index)
    except Exception as e:
        print("skip lib artifact", fn, e)
print("recoverable quarantined signals:", list(lib_sigs.keys()))
for fid, s in signals.items():
    best, best_f = 0.0, None
    for lf, ls in lib_sigs.items():
        lic = ic_series(ls, fr, min_valid=8)
        pair = pd.concat([ics[fid].rename("a"), lic.rename("b")], axis=1).dropna()
        if len(pair) >= 30:
            r = pair["a"].corr(pair["b"])
            if np.isfinite(r) and abs(float(r)) > best:
                best, best_f = abs(float(r)), lf
    print(f"  {fid}: max_abs_lib_corr={best:.4f} (vs {best_f})")

print("\n=== signal-level raw-value correlation among candidates ===")
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        pair = pd.concat([signals[names[i]].stack().rename("a"), signals[names[j]].stack().rename("b")], axis=1).dropna()
        r = pair["a"].corr(pair["b"])
        print(f"  {names[i]} vs {names[j]}: rho={r:.4f} (n={len(pair)})")

# coverage summary
print("\n=== coverage: dates with >=8 valid instruments ===")
for fid, s in signals.items():
    ge8 = s.dropna(thresh=8)
    print(f"  {fid}: {len(ge8)}/{len(s)} dates, assets always valid: {int(s.notna().all().sum())}/15")
