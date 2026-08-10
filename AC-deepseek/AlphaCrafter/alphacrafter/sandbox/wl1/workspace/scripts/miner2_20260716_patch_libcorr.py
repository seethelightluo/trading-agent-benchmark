"""Patch miner2_20260716_lib_corr_update.py:
- handle the miner2_20260716_ id prefix (new persistence)
- verify each persisted signal_artifact round-trips to the reconstructed panel
- recompute pairwise rho from real panels and write provenance into each file"""
import re

path = "scripts/miner2_20260716_lib_corr_update.py"
src = open(path).read()

# 1) panel_key: accept both 20260715 and 20260716 prefixes
src = src.replace(
    """    if fname.startswith('miner2_'):
        return fid.replace('miner2_20260715_', '')""",
    """    if fname.startswith('miner2_'):
        return fid.replace('miner2_20260715_', '').replace('miner2_20260716_', '')""",
)

# 2) artifact verification + rho from artifact cross-check, inserted before pairwise loop
verify_block = '''

# ---- verify embedded signal artifacts round-trip to reconstructed panels ----
import base64, gzip
for f, fid, pk in mapping:
    d = json.load(open(f))
    art = d.get('signal_artifact')
    if art is None:
        print(f"!! NO ARTIFACT {f} (id={fid})")
        continue
    raw = gzip.decompress(base64.b64decode(art['data_b64']))
    M = np.frombuffer(raw, dtype=np.float32).reshape(art['n_dates'], art['n_symbols'])
    syms = art['symbols']
    recon = panels[pk].reindex(idx)
    recon = recon[[c for c in syms if c in recon.columns]]
    A = M
    B = recon.values.astype(np.float32)
    assert A.shape == B.shape, f"shape mismatch {f} {A.shape} vs {B.shape}"
    both = np.isfinite(A) & np.isfinite(B)
    dmax = float(np.abs(A[both] - B[both]).max()) if both.any() else np.nan
    print(f"  artifact-check {fid:36s} max|diff|={dmax:.2e} n_finite={int(both.sum())}")

# ---- pairwise correlation matrix (signed) ----
'''
src = src.replace(
    "\n# pairwise correlation matrix (signed)",
    verify_block + "\n# pairwise correlation matrix (signed)",
    1,
)
open(path, "w").write(src)
print("patched", path)
import py_compile
py_compile.compile(path, doraise=True)
print("syntax OK")
