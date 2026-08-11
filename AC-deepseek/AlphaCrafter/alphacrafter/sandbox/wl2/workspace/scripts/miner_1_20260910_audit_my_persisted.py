"""miner_1 2026-09-10: audit my persisted/effective factors with the deterministic
post-Miner gate (time-averaged daily cross-sectional Spearman rho vs KEPT library).
Admission: IC/ICIR gate pass AND kept_max_rho < 0.5.
"""
import sys, json, glob, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd

GATE_IC = 0.0070
GATE_ICIR = 0.0840

# --- kept library ids: JSON present in factors/ with EFFECTIVE status ---
KEPT = set()
for f in sorted(glob.glob("factors/*.json")):
    if ".bak" in f or "ensemble" in f:
        continue
    try:
        d = json.load(open(f))
        if d.get("validation", {}).get("status") == "EFFECTIVE":
            KEPT.add(d["factor_id"])
    except Exception:
        pass
print("kept ids (%d):" % len(KEPT), sorted(KEPT))

ARTIFACTS = sorted(glob.glob("factors/*.signal.npy"))


def mean_daily_rho(a, b, min_assets=8):
    rows = min(a.shape[0], b.shape[0])
    rhos = []
    for t in range(rows):
        x, y = a[t], b[t]
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < min_assets:
            continue
        xs = pd.Series(x[ok]).rank()
        ys = pd.Series(y[ok]).rank()
        c = xs.corr(ys)
        if np.isfinite(c):
            rhos.append(c)
    return float(np.mean(rhos)) if rhos else 0.0


def rho_table(mat):
    out = {}
    for f in ARTIFACTS:
        fid = os.path.basename(f).replace(".signal.npy", "")
        arr = np.load(f, allow_pickle=True)
        out[fid] = mean_daily_rho(mat, arr)
    mx = max(out.items(), key=lambda kv: abs(kv[1]))
    kept_max = max((abs(v) for k, v in out.items() if k in KEPT), default=0.0)
    return out, mx, kept_max


# my persisted factors to audit
mine = ["zsco_20", "zsco_40", "vol_zscore_20", "accel_mom_20x20", "trend_tstat_20"]
# load IC/ICIR from their JSONs where available
meta = {}
for f in sorted(glob.glob("factors/*.json")):
    if ".bak" in f or "ensemble" in f:
        continue
    try:
        d = json.load(open(f))
        fid = d.get("factor_id")
        if fid in mine:
            m = d.get("validation", {}).get("metrics", {})
            meta[fid] = (m.get("ic_10d"), m.get("icir_10d"), d.get("validation", {}).get("status"))
    except Exception:
        pass

print("\n=== AUDIT of my persisted/candidate factors ===")
for fid in mine:
    path = f"factors/{fid}.signal.npy"
    if not os.path.exists(path):
        print(f"{fid}: NO ARTIFACT -> quarantine risk (no recoverable signal)")
        continue
    mat = np.load(path, allow_pickle=True)
    out, mx, kept_max = rho_table(mat)
    ic, icir, status = meta.get(fid, (None, None, None))
    ok_ic = (ic is not None and abs(ic) >= GATE_IC and abs(icir) >= GATE_ICIR)
    ok_corr = kept_max < 0.5
    print(f"{fid}: ic={ic} icir={icir} status={status} | kept_max_rho={kept_max:.4f} "
          f"all_max_rho={abs(mx[1]):.4f} ({mx[0]}) | GATE_IC={'PASS' if ok_ic else 'FAIL'} "
          f"GATE_RHO={'PASS' if ok_corr else 'FAIL'} ADMIT={ok_ic and ok_corr}")
    # show top-5 abs rho
    top = sorted(out.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    for k, v in top:
        tag = "KEPT" if k in KEPT else "evicted"
        print(f"    {k:28s} rho={v:+.4f} [{tag}]")
