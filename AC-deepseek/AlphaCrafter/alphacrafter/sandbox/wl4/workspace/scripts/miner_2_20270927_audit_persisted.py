"""miner_2 2027-09-27: independent audit of persisted factor files (v2 - dual artifact formats).
1) Reload every factor JSON from disk, verify JSON validity / ids / status / gates.
2) Decode signal artifacts (base64:zlib:csv OR panel_json_v1) and recompute pooled
   pairwise |corr| of the two new EFFECTIVE factors vs the remaining library.
"""
import json, base64, zlib, hashlib
import numpy as np

FILES = [
    "factors/vol_ratio_20_60.json",
    "factors/volume_z_20.json",
    "factors/eurusd_beta_60d_deprecated.json",
    "factors/vol_price_corr_20.json",
    "factors/dn_mkt_beta_60d.json",
    "factors/rate_beta_cn10y_60d.json",
]

IC_TH, ICIR_TH = 0.0070, 0.0840

def decode_artifact(sa):
    if sa.get("format") == "base64:zlib:csv":
        comp = base64.b64decode(sa["data"])
        assert hashlib.sha256(comp).hexdigest()[:16] == sa["sha256"], "sha mismatch"
        csvb = zlib.decompress(comp).decode()
        lines = csvb.split("\n")
        cols = lines[0].split(",")
        panel = {}
        for ln in lines[1:]:
            if not ln.strip():
                continue
            parts = ln.split(",")
            d = parts[0]
            for c, v in zip(cols[1:], parts[1:]):
                if v != "":
                    panel[(d, c)] = float(v)
        return panel
    if sa.get("format") == "panel_json_v1":
        panel = {}
        dates, assets = sa["dates"], sa["assets"]
        vals = sa["values"]
        for a, arr in vals.items():
            for d, v in zip(dates, arr):
                if v is not None:
                    panel[(d, a)] = float(v)
        return panel
    raise ValueError(f"unknown artifact format {sa.get('format')}")

print("=== reload & verify persisted files ===")
panels = {}
for f in FILES:
    d = json.load(open(f))
    fid = d["factor_id"]
    st = d["validation"]["status"]
    m = d["validation"]["metrics"]
    ic, icir = m["ic"], m["icir"]
    sa = d["validation"]["signal_artifact"]
    panel = decode_artifact(sa)
    panels[fid] = panel
    ok = True
    if st == "EFFECTIVE":
        if not (abs(ic) >= IC_TH and abs(icir) >= ICIR_TH):
            ok = False
    print(f"  {fid:26s} status={st:11s} IC={ic:+.4f} ICIR={icir:+.4f} "
          f"n_valid={len(panel)} fmt={sa.get('format')} gate_ok={ok}")

print()
print("=== pooled pairwise |corr| of new factors vs library (from artifacts) ===")
lib_ids = ["vol_price_corr_20", "dn_mkt_beta_60d", "rate_beta_cn10y_60d", "eurusd_beta_60d"]
for c in ["vol_ratio_20_60", "volume_z_20"]:
    best, bestr = None, 0.0
    for l in lib_ids:
        keys = sorted(set(panels[c]) & set(panels[l]))
        if len(keys) < 30:
            print(f"  {c:20s} vs {l:22s} insufficient overlap n={len(keys)}")
            continue
        va = np.array([panels[c][k] for k in keys])
        vb = np.array([panels[l][k] for k in keys])
        r = abs(np.corrcoef(va, vb)[0, 1])
        print(f"  {c:20s} vs {l:22s} rho={r:.3f} n={len(keys)}")
        if r > bestr:
            bestr, best = r, l
    print(f"  -> {c}: max_abs_library_correlation={bestr:.3f} vs {best}")

print()
print("AUDIT COMPLETE")
