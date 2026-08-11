"""Verify persisted factor JSONs are valid, reloadable, and internally consistent.
Handles both gzip-b64 artifact format and normalized plain panel_json_v1."""
import json, base64, gzip, sys
from pathlib import Path
import numpy as np

fids = ["rate_beta_cn10y_60d", "eurusd_beta_60d", "dn_mkt_beta_60d"]
ok = True


def extract_panel(rec):
    art = rec["validation"]["signal_artifact"]
    if "payload_b64_gzip" in art:
        payload = json.loads(gzip.decompress(base64.b64decode(art["payload_b64_gzip"])))
        return payload["dates"], payload["assets"], payload["values"], art["format"]
    return art["dates"], art["assets"], art["values"], art["format"]


for fid in fids:
    p = Path(f"factors/{fid}.json")
    try:
        rec = json.loads(p.read_text())
    except Exception as e:
        print(f"[FAIL] {fid}: JSON parse error: {e}")
        ok = False
        continue
    v = rec.get("validation", {})
    m = v.get("metrics", {})
    art = v.get("signal_artifact", {})
    try:
        dates, assets, values, fmt = extract_panel(rec)
        n_dates = len(dates)
        n_assets = len(assets)
        n_vals = sum(1 for a in assets for x in values[a] if x is not None)
    except Exception as e:
        print(f"[FAIL] {fid}: artifact decode error: {e}")
        ok = False
        continue
    checks = {
        "factor_id": rec.get("factor_id") == fid,
        "status_effective": v.get("status") == "EFFECTIVE",
        "ic_gate": abs(m.get("ic", 0)) >= 0.007,
        "icir_gate": abs(m.get("icir", 0)) >= 0.084,
        "artifact_dims": (art.get("n_dates") == n_dates) and (art.get("n_assets") == n_assets),
        "artifact_values": n_vals > 1000,
        "libcorr_present": "max_abs_library_correlation" in m,
        "last_validated": v.get("last_validated"),
    }
    status = "OK" if all(checks.values()) and checks["last_validated"] else "MISMATCH"
    if status != "OK":
        ok = False
    print(f"[{status}] {fid}: id={checks['factor_id']} status={checks['status_effective']} "
          f"ic_gate={checks['ic_gate']} icir_gate={checks['icir_gate']} "
          f"art_dims={checks['artifact_dims']} art_vals={checks['artifact_values']} "
          f"libcorr={m.get('max_abs_library_correlation')} validated={checks['last_validated']} "
          f"artifact={fmt} n_dates={n_dates} n_assets={n_assets}")

from itertools import combinations


def load_signal(fid):
    rec = json.loads(Path(f"factors/{fid}.json").read_text())
    return extract_panel(rec)


print("\nPairwise cross-sectional rank correlations (new trio):")
for a, b in combinations(fids, 2):
    d1, s1, v1, _ = load_signal(a)
    d2, s2, v2, _ = load_signal(b)
    common_dates = sorted(set(d1) & set(d2))
    common_assets = [x for x in s1 if x in s2]
    rhos = []
    for d in common_dates:
        i1, i2 = d1.index(d), d2.index(d)
        x = [v1[x][i1] for x in common_assets if v1[x][i1] is not None and v2[x][i2] is not None]
        y = [v2[x][i2] for x in common_assets if v1[x][i1] is not None and v2[x][i2] is not None]
        if len(x) >= 5:
            r = np.corrcoef(x, y)[0, 1]
            if not np.isnan(r):
                rhos.append(r)
    print(f"  {a} vs {b}: mean rho={np.mean(rhos):.3f} over {len(rhos)} dates")

print("\nALL PERSISTENCE CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
