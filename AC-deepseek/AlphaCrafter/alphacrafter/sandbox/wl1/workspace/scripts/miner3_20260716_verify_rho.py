"""miner_3: verify pooled-rho mechanics of the post-Miner gate and test
whether cross-sectional normalization kills the common-level correlation."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

panel = pd.read_pickle("scripts/panel_cache.pkl")
C = panel["close"].astype(float)
O = panel["open"].astype(float)
H = panel["high"].astype(float)
L = panel["low"].astype(float)
V = panel["vol"].astype(float)
SYMS = list(C.columns)
idx = C.index
LRET = np.log(C / C.shift(1))
RET = C.pct_change()

# ---------- reconstruct library signals ----------
lib = {}
lib["cz_rev1"] = -LRET.sub(LRET.mean(axis=1), axis=0).div(LRET.std(axis=1) + 1e-12, axis=0)
pk20 = np.sqrt((np.log(H / L) ** 2).rolling(20).mean() / (4 * np.log(2)))
lib["rev1_pk"] = -LRET / (pk20 + 1e-12)
ineff = (C / C.shift(20) - 1.0).abs() / (LRET.abs().rolling(20).sum() + 1e-12)
lib["rev1_x_inveff"] = LRET * (1 - ineff)
lib["id_rev_1d"] = -(C / O - 1.0)
lib["nbody_1d"] = -(C - O) / (H - L + 1e-12)
for k in (1, 2, 3, 5):
    lib[f"nclv_{k}d"] = -(C - L.rolling(k).min()) / (H.rolling(k).max() - L.rolling(k).min() + 1e-12)
lib["rev_1d"] = -LRET
lib["rev_2d"] = -(np.log(C) - np.log(C.shift(2)))
lib["rev_3d"] = -(np.log(C) - np.log(C.shift(3)))
lib["rev_1d_vs"] = -LRET / (RET.rolling(20).std() + 1e-12)
lib["mom_10d_skip5"] = np.log(C / C.shift(10)) - np.log(C / C.shift(5))

# real artifacts
art = {}
for f in ["miner1_20260716_er20", "miner1_20260716_rev5x_er_soft", "miner2_20260716_mom_10d_skip5", "miner2_20260716_nclv_1d"]:
    a = np.load(f"factors/{f}.npy", allow_pickle=True).astype(float)
    art[f] = a


def pooled_rho(a, b, min_n=500):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    m = (~np.isnan(a)) & (~np.isnan(b))
    if m.sum() < min_n:
        return float("nan")
    return float(spearmanr(a[m], b[m]).statistic)


lib_np = {k: v.to_numpy(dtype=float) for k, v in lib.items()}

# verify reconstruction vs artifacts
print("=== recon vs artifact check ===")
print("mom_10d_skip5:", pooled_rho(lib_np["mom_10d_skip5"], art["miner2_20260716_mom_10d_skip5"]))
print("nclv_1d      :", pooled_rho(lib_np["nclv_1d"], art["miner2_20260716_nclv_1d"]))

print("\n=== intra-library pooled rho matrix (reconstructed) ===")
names = list(lib.keys())
for i in range(len(names)):
    row = []
    for j in range(len(names)):
        r = pooled_rho(lib_np[names[i]], lib_np[names[j]])
        row.append(f"{r:+.2f}" if np.isfinite(r) else "  NA")
    print(f"{names[i]:14s}", " ".join(row))

# test: cross-sectional demean of mom_10d_skip5 vs raw
mom = lib_np["mom_10d_skip5"]
df = pd.DataFrame(mom, index=idx, columns=SYMS)
mom_cs = (df.sub(df.mean(axis=1), axis=0)).to_numpy()
mom_cs_z = (df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-12, axis=0)).to_numpy()
print("\n=== CS-demean effect on rho vs raw library ===")
print("mom raw   vs raw mom :", pooled_rho(mom, mom))
print("mom csd   vs raw mom :", pooled_rho(mom_cs, mom))
print("mom csdz  vs raw mom :", pooled_rho(mom_cs_z, mom))
print("mom csdz  vs nclv_1d :", pooled_rho(mom_cs_z, lib_np["nclv_1d"]))
print("mom csdz  vs rev_1d  :", pooled_rho(mom_cs_z, lib_np["rev_1d"]))
# rsi-2 style factor
def rsi2(C, n=2):
    chg = C.diff()
    up = chg.clip(lower=0).rolling(n).mean()
    dn = (-chg.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))
r2 = rsi2(C).to_numpy()
print("rsi2 raw vs nclv_1d:", pooled_rho(r2, lib_np["nclv_1d"]))
r2df = pd.DataFrame(r2, index=idx, columns=SYMS)
r2cs = r2df.sub(r2df.mean(axis=1), axis=0).div(r2df.std(axis=1) + 1e-12, axis=0).to_numpy()
print("rsi2 csd vs nclv_1d:", pooled_rho(r2cs, lib_np["nclv_1d"]))
print("rsi2 csd vs mom10d :", pooled_rho(r2cs, mom))
