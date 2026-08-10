"""Patch miner2_20260716_persist_reversal.py to embed a recoverable signal
artifact (gzip+base64 float32 daily panel, NaN preserved) into every persisted
factor JSON, so the deterministic post-Miner gate can recompute pairwise rho
from real signal data instead of quarantining for missing artifacts."""
import re

path = "scripts/miner2_20260716_persist_reversal.py"
src = open(path).read()

# 1) imports
src = src.replace(
    "import sys, os, json, time",
    "import sys, os, json, time, base64, gzip",
)

# 2) artifact builder before Part D
artifact_fn = '''

def make_artifact(panel):
    """Recoverable signal artifact: gzip+base64 of float32 (dates x symbols)
    matrix on the validation index, NaN bit patterns preserved."""
    P = panel.reindex(idx).astype(np.float32)
    cols = [c for c in SYMBOLS if c in P.columns]
    M = P[cols].values.astype(np.float32, copy=False)
    b64 = base64.b64encode(gzip.compress(M.tobytes(), compresslevel=6)).decode("ascii")
    return {
        "format": "gzip+base64 float32 matrix (dates x symbols), NaN preserved",
        "symbols": cols,
        "n_dates": int(M.shape[0]),
        "n_symbols": int(M.shape[1]),
        "date_start": str(idx[0].date()),
        "date_end": str(idx[-1].date()),
        "data_b64": b64,
        "recovery": "base64.b64decode -> gzip.decompress -> np.frombuffer(dtype=float32).reshape(n_dates, n_symbols)",
    }

# ============ Part D: persist ============'''
src = src.replace("\n# ============ Part D: persist ============", artifact_fn, 1)

# 3) inject signal_artifact into each doc before json.dump
src = src.replace(
    '''    path = f"factors/{factor_id}.json"
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)''',
    '''    doc["signal_artifact"] = make_artifact(r["panel"])
    path = f"factors/{factor_id}.json"
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)''',
)

# 4) reload-verify inside persist loop
src = src.replace(
    '''    persisted.append(path)
    print(f"[persisted] {path}")''',
    '''    chk = json.load(open(path))
    assert chk["factor_id"] == factor_id, f"id mismatch {path}"
    assert chk["validation"]["status"] == "EFFECTIVE"
    art = chk.get("signal_artifact")
    assert art is not None and len(art["data_b64"]) > 1000, f"no artifact {path}"
    assert art["n_symbols"] == 15 and art["n_dates"] == len(idx), f"artifact shape {path}"
    persisted.append(path)
    print(f"[persisted+verified] {path} artifact={art['n_dates']}x{art['n_symbols']} b64len={len(art['data_b64'])}")''',
)

open(path, "w").write(src)
print("patched", path)
# quick syntax check
import py_compile
py_compile.compile(path, doraise=True)
print("syntax OK")
