"""Screener: correlation gate on as-consumed transformed signals (neutral-fill -> CS rank -> z-score)."""
import json, numpy as np, os
from scipy.stats import rankdata

ens = json.load(open("factors/factor_ensemble.json"))
sel = ens["selected_factors"]

def load_signal(f):
    p = f["signal_artifact"]
    if not os.path.exists(p):
        p = "factors/" + f["signal_artifact"].split("/")[-1]
    return np.load(p)

def transform(a):
    """neutral-fill NaN with CS median, then CS rank -> z-score per date row."""
    out = a.copy().astype(float)
    for i in range(out.shape[0]):
        row = out[i]
        m = ~np.isnan(row)
        if m.sum() == 0:
            out[i] = 0.0
        else:
            med = np.nanmedian(row)
            row[~m] = med
            r = rankdata(row) / (len(row) + 1)
            r = (r - r.mean()) / (r.std() + 1e-12)
            out[i] = r
    return out

names, T = [], []
for f in sel:
    names.append(f["factor_id"])
    T.append(transform(load_signal(f)))

def pair_corr(ti, tj, rows):
    a, b = ti[rows], tj[rows]
    m = ~(np.isnan(a) | np.isnan(b))
    return np.corrcoef(a[m], b[m])[0, 1]

for label, rows in [("full", slice(None)), ("last250", slice(-250, None)), ("last60", slice(-60, None))]:
    mx, pair = 0.0, None
    for i in range(len(T)):
        for j in range(i + 1, len(T)):
            rho = pair_corr(T[i], T[j], rows)
            if abs(rho) > mx:
                mx, pair = abs(rho), (names[i], names[j], rho)
    print(f"{label:8s} max |rho| = {mx:.4f}  pair: {pair[0]} vs {pair[1]} (rho={pair[2]:+.4f}) | gate<0.7: {mx < 0.7}")

# direction-adjusted exposure correlation (what matters for portfolio redundancy)
print("\nDirection-adjusted exposure corr (dir_i*dir_j*rho), last250:")
dirs = {f["factor_id"]: f["direction"] for f in sel}
for i in range(len(T)):
    for j in range(i + 1, len(T)):
        rho = pair_corr(T[i], T[j], slice(-250, None))
        expo = dirs[names[i]] * dirs[names[j]] * rho
        if abs(expo) > 0.5:
            print(f"  {names[i]} vs {names[j]}: raw={rho:+.4f} exposure={expo:+.4f}")
