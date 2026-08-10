"""Screener: verify ensemble math + pairwise correlation gate from signal artifacts."""
import json, numpy as np, os

ens = json.load(open("factors/factor_ensemble.json"))
sel = ens["selected_factors"]
print("selected count:", len(sel), "(cap 10)")

# 1) weights recompute: q = |IC|*|ICIR|, w = q/sum(q), dir = sign(IC)
qs, ws = [], []
for f in sel:
    q = abs(f["ic"]) * abs(f["icir"])
    qs.append(q)
    ws.append(q)
    print(f"{f['factor_id']:26s} ic={f['ic']:+.4f} icir={f['icir']:+.4f} q={q:.6f} "
          f"w_json={f['weight']:.6f} dir_json={f['direction']}")
tot = sum(qs)
print("sum q = %.8f" % tot)
ok = True
for f, q in zip(sel, qs):
    w = q / tot
    d = 1 if f["ic"] > 0 else -1
    if abs(w - f["weight"]) > 1e-9 or d != f["direction"]:
        ok = False
        print("MISMATCH", f["factor_id"], w, f["weight"], d, f["direction"])
print("weights sum:", sum(f["weight"] for f in sel), "| recompute OK:", ok, "| all non-neg:", all(f["weight"] >= 0 for f in sel))

# 2) signal artifacts exist + pairwise corr gate
names, arrs = [], []
for f in sel:
    p = f["signal_artifact"]
    if not os.path.exists(p):
        p = "factors/" + f["signal_artifact"].split("/")[-1]
    a = np.load(p)
    names.append(f["factor_id"]); arrs.append(a)
    print(f"artifact {f['factor_id']}: shape={a.shape} nan={np.isnan(a).mean():.3f}")

# pairwise spearman-ish (pearson on rank) on overlapping non-nan cells
K = len(arrs)
maxrho, pair = 0.0, None
for i in range(K):
    for j in range(i + 1, K):
        a, b = arrs[i], arrs[j]
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < 100:
            continue
        from scipy.stats import rankdata
        ra = rankdata(a[m]); rb = rankdata(b[m])
        rho = np.corrcoef(ra, rb)[0, 1]
        if abs(rho) > maxrho:
            maxrho, pair = abs(rho), (names[i], names[j], rho)
print(f"max pairwise |rho| = {maxrho:.4f} ({pair[0]} vs {pair[1]}, rho={pair[2]:+.4f}) -> gate <0.7: {maxrho < 0.7}")
