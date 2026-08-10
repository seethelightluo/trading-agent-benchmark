"""Reload active library, extract metrics, recompute as-consumed pairwise correlations."""
import json
import numpy as np
from pathlib import Path

BASE = Path("factors")

def list_active():
    out = []
    for p in sorted(BASE.glob("*.json")):
        if p.name == "factor_ensemble.json" or p.name.endswith(".bak"):
            continue
        try:
            j = json.load(open(p))
        except Exception as e:
            print("skip", p, e)
            continue
        if j.get("factor_id"):
            out.append(p)
    return out

active = list_active()
print("=== ACTIVE FACTOR FILES ===")
for p in active:
    print(" ", p)

rows = []
for p in active:
    j = json.load(open(p))
    v = j.get("validation", {})
    m = v.get("metrics", {})
    rows.append({
        "fid": j["factor_id"],
        "status": v.get("status", "?"),
        "ic": m.get("ic"),
        "icir": m.get("icir"),
        "hit": m.get("ic_hit_ratio"),
        "turn": m.get("turnover_10d_rank"),
        "cov_asset": m.get("coverage_asset_days"),
        "cov_date": m.get("coverage_dates_ge8"),
        "maxlib": m.get("max_abs_library_correlation"),
        "last_validated": v.get("last_validated"),
        "artifact": j.get("signal_artifact"),
        "expected_dir": j.get("expected_direction"),
    })

print("\n=== METRICS TABLE ===")
print(f"{'factor':26s} {'status':10s} {'IC':>7s} {'ICIR':>7s} {'q':>9s} {'hit':>5s} {'turn':>5s} {'covA':>5s} {'covD':>5s} {'maxLib':>7s} {'dir':>4s}")
for r in rows:
    if r["ic"] is None:
        continue
    q = abs(r["ic"]) * abs(r["icir"])
    print(f"{r['fid']:26s} {r['status']:10s} {r['ic']:7.4f} {r['icir']:7.4f} {q:9.6f} "
          f"{r['hit']:5.3f} {r['turn']:5.3f} {r['cov_asset']:5.3f} {r['cov_date']:5.3f} "
          f"{r['maxlib']:7.4f} {r['expected_dir']:>4d}")

# load signals
def load_signal(fid):
    p = BASE / f"{fid}.signal.npy"
    if not p.exists():
        return None
    return np.load(p)

def transform(x):
    out = np.full_like(x, np.nan, dtype=float)
    for t in range(x.shape[0]):
        row = x[t]
        m = ~np.isnan(row)
        if m.sum() == 0:
            continue
        filled = row.copy()
        filled[~m] = np.nanmedian(row[m])
        ranks = np.argsort(np.argsort(filled))
        pct = ranks / (len(filled) - 1) if len(filled) > 1 else 0.5
        z = (pct - 0.5) * np.sqrt(12.0)
        out[t] = np.clip(z, -3.0, 3.0)
    return out

def spearman(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 5:
        return np.nan
    ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else np.nan

fids = [r["fid"] for r in rows if r["ic"] is not None]
print("\n=== SIGNAL ARTIFACT AVAILABILITY ===")
sig = {}
for fid in fids:
    s = load_signal(fid)
    if s is None:
        print(f"  {fid:26s} NO ARTIFACT")
    else:
        print(f"  {fid:26s} shape={s.shape} nan={np.isnan(s).sum()} lastrow_nan={np.sum(np.isnan(s[-1]))}/15")
        sig[fid] = s

print("\n=== AS-CONSUMED PAIRWISE SPEARMAN (full) ===")
tr = {fid: transform(s) for fid, s in sig.items()}
fids_sig = list(sig.keys())
for i in range(len(fids_sig)):
    for j in range(i + 1, len(fids_sig)):
        a, b = fids_sig[i], fids_sig[j]
        rho_full = spearman(tr[a], tr[b])
        rho_250 = spearman(tr[a][-250:], tr[b][-250:])
        rho_60 = spearman(tr[a][-60:], tr[b][-60:])
        flag = "OK" if abs(rho_full) < 0.5 and abs(rho_250) < 0.5 and abs(rho_60) < 0.5 else "CHECK"
        print(f"  {a:26s} vs {b:26s} full={rho_full:+.4f} last250={rho_250:+.4f} last60={rho_60:+.4f} [{flag}]")
