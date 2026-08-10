"""miner_1: Persist btc_beta_60 (gate-passing candidate from explore_v3) with recoverable signal artifact.

btc_beta_60 passed the shared admission gate (|IC10| >= 0.0070, |ICIR10| >= 0.0840) in
scripts/miner_1_20260730_explore_v3.py but was NOT written to factors/. This script:
1. recomputes the signal + metrics,
2. verifies gates,
3. computes max-abs IC-series correlation vs the 4 quarantined factors that have
   recoverable signal artifacts (breadth_cond_mom_20, dxy_beta_60, mom_vol_scaled_20x10,
   skew_vol_comp_20),
4. embeds the full signal panel as base64:zlib:csv artifact,
5. writes factors/btc_beta_60.json, reads back and verifies.
"""
import sys, json, os, base64, zlib, hashlib, io
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, macro_closes, forward_returns,
                             ic_series, summary_metrics, regime_split, WATCH)

VIS = "2026-07-29"
H = 10
close = closes_panel(VIS)
fr = forward_returns(close, H)
ret = close.pct_change()
btc = close["BTC"]

# --- recompute signal: rolling 60d beta of each asset return vs BTC return ---
def rolling_beta(x, y, w=60, mp=40):
    out = {}
    for a in x.columns:
        pair = pd.concat([x[a].rename("a"), y.rename("y")], axis=1).dropna()
        b = (pair["a"].rolling(w, min_periods=mp).cov(pair["y"])
             / pair["y"].rolling(w, min_periods=mp).var())
        out[a] = b
    return pd.DataFrame(out).reindex(x.index)

sig = rolling_beta(ret, btc.pct_change(), 60, 40)
print(f"signal shape={sig.shape} valid={int(sig.notna().sum().sum())}", flush=True)

ic = ic_series(sig, fr, min_valid=8)
m = summary_metrics(ic, sig, fr, close, h=H)
assert m is not None, "insufficient IC dates"
m["regime"] = regime_split(ic)
gate_ic = abs(m["ic"]) >= 0.0070
gate_icir = abs(m["icir"] or 0) >= 0.0840
print(f"IC10={m['ic']:+.4f} ICIR10={m['icir']:+.3f} hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']} "
      f"cov_ad={m['coverage_asset_days']:.3f} cov_d8={m['coverage_dates_ge8']:.3f} "
      f"turn={m['turnover_10d_rank'] if m['turnover_10d_rank'] is not None else float('nan'):.3f}")
print(f"decay={m['decay_ic_by_horizon']}")
print(f"regimes={ {k: (v['ic'], v['icir'], v['n']) for k, v in m['regime'].items()} }")
assert gate_ic and gate_icir, f"btc_beta_60 fails gates: ic={m['ic']}, icir={m['icir']}"

# --- redundancy check vs quarantined library artifacts (recoverable only) ---
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

rho_map = {}
for lf, ls in lib_sigs.items():
    lic = ic_series(ls, fr, min_valid=8)
    pair = pd.concat([ic.rename("a"), lic.rename("b")], axis=1).dropna()
    if len(pair) >= 30:
        r = pair["a"].corr(pair["b"])
        rho_map[lf] = round(float(r), 4) if np.isfinite(r) else None
best = max([abs(v) for v in rho_map.values() if v is not None], default=0.0)
m["max_abs_library_correlation"] = round(best, 4)
print(f"rho vs library: {rho_map}  max_abs={best:.4f}")
assert best < 0.5, f"redundant with library: max rho {best:.4f}"

# --- signal artifact ---
sig_out = sig.copy()
sig_out.index = sig_out.index.strftime("%Y-%m-%d")
csv_txt = sig_out.to_csv()
raw = csv_txt.encode("utf-8")
compressed = zlib.compress(raw, 9)
b64 = base64.b64encode(compressed).decode("ascii")
artifact = {
    "format": "base64:zlib:csv",
    "description": ("Factor signal panel: rows = dates (YYYY-MM-DD), cols = 15 watchlist symbols. "
                    "Recover with zlib.decompress(base64.b64decode(data)).decode() -> pandas.read_csv(StringIO)."),
    "columns": WATCH,
    "shape": [int(sig_out.shape[0]), int(sig_out.shape[1])],
    "n_valid_values": int(sig_out.notna().sum().sum()),
    "sha256": hashlib.sha256(raw).hexdigest(),
    "data": b64,
}

meta = {
    "factor_id": "btc_beta_60",
    "factor_name": "BTC beta 60d (crypto risk linkage)",
    "version": "1.0.0",
    "calculation": {
        "expression": "beta(pct_change(asset), pct_change(BTC), 60)",
        "description": ("Rolling 60-day regression beta of each tradable asset's daily return on "
                        "BTC daily return. Captures cross-asset crypto risk linkage: assets whose "
                        "returns co-move with BTC (crypto-adjacent beta) tend to persist in relative "
                        "performance over 10d. Positive IC in all three regimes; strongest 2025-26."),
    },
    "dependencies": ["close"],
    "parameters": {"lookback": 60, "min_periods": 40, "benchmark": "BTC", "horizon": 10, "min_valid_assets": 8},
    "expected_direction": 1,
    "validation": {
        "status": "EFFECTIVE",
        "period": f"2020-01-01..{VIS}",
        "admission_horizon": H,
        "last_validated": "2026-07-30",
        "regime_notes": ("Validated on 15-asset cross-asset universe; 629 IC dates with >=8 valid "
                         "instruments. IC by regime: 2020-22 +0.0615 (ICIR 0.156), 2023-24 +0.0311 "
                         "(ICIR 0.077, weaker), 2025-26 +0.1080 (ICIR 0.237, strong). Decay increases "
                         "with horizon (h=20 IC 0.0845), i.e. signal is slow/persistent, not mean-reverting."),
        "metrics": m,
        "signal_artifact": artifact,
    },
    "tags": ["beta", "crypto", "cross-asset", "risk-linkage"],
}

out = "factors/btc_beta_60.json"
with open(out, "w") as fh:
    json.dump(meta, fh, indent=1)
print(f"written {out} ({len(json.dumps(meta))} bytes)", flush=True)

# --- read back & verify ---
with open(out) as fh:
    back = json.load(fh)
assert back["factor_id"] == "btc_beta_60"
assert back["validation"]["status"] == "EFFECTIVE"
assert abs(back["validation"]["metrics"]["ic"]) >= 0.0070
assert abs(back["validation"]["metrics"]["icir"] or 0) >= 0.0840
a = back["validation"]["signal_artifact"]
dec = zlib.decompress(base64.b64decode(a["data"])).decode("utf-8")
sig_back = pd.read_csv(io.StringIO(dec), index_col=0)
sig_back.index = pd.to_datetime(sig_back.index)
assert sig_back.shape == tuple(a["shape"]), (sig_back.shape, a["shape"])
assert hashlib.sha256(dec.encode("utf-8")).hexdigest() == a["sha256"]
rec_ic = ic_series(sig_back.reindex(close.index), fr, min_valid=8)
rec_icir = float(rec_ic.mean() / rec_ic.std(ddof=1)) if len(rec_ic) > 2 and rec_ic.std(ddof=1) > 0 else None
print(f"VERIFY ok: shape={sig_back.shape} sha256 ok recovered IC={rec_ic.mean():+.4f} "
      f"ICIR={rec_icir:+.4f} vs persisted IC={m['ic']:+.4f} ICIR={m['icir']:+.4f}")
print("PERSISTED AND VERIFIED")
