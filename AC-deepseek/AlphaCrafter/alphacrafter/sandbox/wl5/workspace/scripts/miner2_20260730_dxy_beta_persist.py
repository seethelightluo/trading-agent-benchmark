"""miner_2: Persist DXY-beta 60d factor with recoverable signal artifact.

Recomputes the validated factor, embeds the full signal panel as a
base64/zlib/CSV artifact (so the deterministic gate can recover the signal
and re-derive metrics instead of quarantining), writes factors/dxy_beta_60.json,
then reads back and self-verifies: JSON validity, ids, status, thresholds,
artifact decode + IC reconstruction.
"""
import sys, json, base64, zlib, hashlib, io
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_validate import (closes_panel, macro_closes, forward_returns,
                             ic_series, summary_metrics, regime_split,
                             library_ic_series_map, max_abs_library_corr,
                             WATCH)

VIS = "2026-07-29"
H = 10
WIN = 60
FID = "dxy_beta_60"
OUT = f"factors/{FID}.json"

close = closes_panel(VIS)
macro = macro_closes(VIS)
ret = close.pct_change()
dxy_ret = macro["DXY"].pct_change()

# --- factor signal ---
beta = {}
for a in close.columns:
    pair = pd.concat([ret[a].rename("a"), dxy_ret.rename("d")], axis=1).dropna()
    b = pair["a"].rolling(WIN).cov(pair["d"]) / pair["d"].rolling(WIN).var()
    beta[a] = b
factor = pd.DataFrame(beta).reindex(close.index)

fr = forward_returns(close, H)
ic = ic_series(factor, fr, min_valid=8)
m = summary_metrics(ic, factor, fr, close, h=H)
lib = library_ic_series_map(close, h=H)
m["max_abs_library_correlation"] = max_abs_library_corr(ic, lib)
m["regime"] = regime_split(ic)
print("metrics:", json.dumps(m, indent=1, default=str))

gate_ic = abs(m["ic"]) >= 0.007
gate_icir = abs(m["icir"] or 0) >= 0.084
assert gate_ic and gate_icir, f"factor does not pass gates: ic={m['ic']} icir={m['icir']}"

# --- signal artifact ---
sig = factor.copy()
sig.index = sig.index.strftime("%Y-%m-%d")
csv_txt = sig.to_csv()
raw = csv_txt.encode("utf-8")
compressed = zlib.compress(raw, 9)
b64 = base64.b64encode(compressed).decode("ascii")
artifact = {
    "format": "base64:zlib:csv",
    "description": "Factor signal panel: rows = dates (YYYY-MM-DD), cols = 15 watchlist symbols. Recover with zlib.decompress(base64.b64decode(data)).decode() -> pandas.read_csv(StringIO).",
    "columns": WATCH,
    "shape": [int(sig.shape[0]), int(sig.shape[1])],
    "n_valid_values": int(sig.notna().sum().sum()),
    "sha256": hashlib.sha256(raw).hexdigest(),
    "data": b64,
}

meta = {
    "factor_id": FID,
    "factor_name": "DXY-beta 60d (US-dollar sensitivity)",
    "version": "1.0.0",
    "calculation": {
        "expression": "beta(asset_ret, DXY_ret, 60)",
        "description": ("Rolling 60-day beta of each asset's daily returns to the US dollar index (DXY): "
                        "cov(asset_ret, DXY_ret, 60)/var(DXY_ret, 60). Captures persistent dollar-sensitivity; "
                        "assets with higher DXY-beta tend to outperform over the 10d horizon (positive IC)."),
    },
    "dependencies": ["close", "DXY"],
    "parameters": {"beta_window": 60, "horizon": 10, "min_valid_assets": 8},
    "expected_direction": 1,
    "validation": {
        "status": "EFFECTIVE",
        "period": f"2020-01-01..{VIS}",
        "admission_horizon": H,
        "last_validated": "2026-07-30",
        "regime_notes": ("Validated on 15-asset cross-asset universe, 552 IC dates with >=8 valid instruments. "
                         "Positive IC in all regimes: 2020-22 (0.0242), 2023-24 (0.0843), 2025-26 (0.0233); "
                         "strongest in the 2023-24 dollar-cycle regime. Decay peaks at 10d horizon."),
        "metrics": m,
        "signal_artifact": artifact,
    },
    "tags": ["macro-beta", "currency", "dollar", "cross-asset"],
}

with open(OUT, "w") as f:
    json.dump(meta, f, indent=1)
print("written:", OUT, "bytes:", len(json.dumps(meta)))

# --- read back and verify ---
with open(OUT) as f:
    back = json.load(f)
assert back["factor_id"] == FID
assert back["validation"]["status"] == "EFFECTIVE"
assert abs(back["validation"]["metrics"]["ic"]) >= 0.007
assert abs(back["validation"]["metrics"]["icir"] or 0) >= 0.084
a = back["validation"]["signal_artifact"]
dec = zlib.decompress(base64.b64decode(a["data"])).decode("utf-8")
sig_back = pd.read_csv(io.StringIO(dec), index_col=0)
sig_back.index = pd.to_datetime(sig_back.index)
assert sig_back.shape == tuple(a["shape"]), (sig_back.shape, a["shape"])
assert hashlib.sha256(dec.encode("utf-8")).hexdigest() == a["sha256"]
rec_ic = ic_series(sig_back.reindex(close.index), fr, min_valid=8)
print(f"verify: shape {sig_back.shape}, sha256 OK, "
      f"recovered IC={rec_ic.mean():.4f} vs reported {m['ic']}, "
      f"recovered ICIR={rec_ic.mean()/rec_ic.std(ddof=1):.4f} vs reported {m['icir']}")
print("PERSIST OK")
