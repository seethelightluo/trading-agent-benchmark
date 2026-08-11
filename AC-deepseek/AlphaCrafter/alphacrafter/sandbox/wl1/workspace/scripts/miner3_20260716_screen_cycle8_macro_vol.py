"""miner_3 cycle-8 screening: macro-sensitivity, volume, and yield-beta factor families.

Context:
- Library (14 distinct effective factors) is dominated by 1-5d mean-reversion +
  mom_10d_skip5 (used negatively). Prior miner3 reversal/vol candidates were
  evicted for pooled rho > 0.5 vs mom_10d_skip5.
- This cycle focuses on families with NO prior library representation:
  A) rolling sensitivity (beta/corr) of each asset to observation-only macro
     series (DXY, USDJPY, EURUSD, USDCNY, VIX)
  B) volume-based signals (log-vol z, vol trend, Amihud illiquidity)
  C) yield-curve sensitivity (US10Y / CN10Y beta)
  D) macro-conditioned intraday reversal (VIX/DXY regime) - re-check rho
  E) cross-sectional deviation / dispersion
Gate: |IC1|>=0.007, |ICIR1|>=0.084 on 2021-01-04..2026-07-15, rho_max<0.5.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

EVAL_START = pd.Timestamp("2021-01-04")
END = pd.Timestamp("2026-07-15")
GATE_IC, GATE_ICIR = 0.0070, 0.0840

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
fwd1 = RET.shift(-1)
ev_idx = idx[(idx >= EVAL_START) & (idx <= END)]
print(f"panel {C.shape} idx {idx.min().date()}..{idx.max().date()}, eval dates={len(ev_idx)}")

# ---------------- library reconstruction (14 effective factors) ----------------
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
lib_np = {k: v.to_numpy(dtype=float) for k, v in lib.items()}


def pooled_rho(a, b, min_n=500):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    m = (~np.isnan(a)) & (~np.isnan(b))
    if m.sum() < min_n:
        return float("nan")
    return float(spearmanr(a[m], b[m]).statistic)


# ---------------- helpers ----------------
def zscore_win(X, win=250, minp=60):
    df = pd.DataFrame(X, index=idx, columns=SYMS)
    mu = df.rolling(win, min_periods=minp).mean()
    sd = df.rolling(win, min_periods=minp).std()
    return ((df - mu) / (sd + 1e-12)).to_numpy(dtype=float)


def zscore_full(X):
    df = pd.DataFrame(X, index=idx, columns=SYMS)
    return df.sub(df.mean()).div(df.std() + 1e-12).to_numpy(dtype=float)


def roll_beta(x_ret, m_ret, win=60, minp=30):
    """rolling beta of each x_ret column to m_ret (Series); returns ndarray (n, ncol)"""
    out = pd.DataFrame(index=x_ret.index, columns=x_ret.columns, dtype=float)
    var = m_ret.rolling(win, min_periods=minp).var()
    for s in x_ret.columns:
        cov = x_ret[s].rolling(win, min_periods=minp).cov(m_ret)
        out[s] = cov / (var + 1e-12)
    return out.to_numpy(dtype=float)


def roll_corr(x_ret, m_ret, win=60, minp=30):
    out = pd.DataFrame(index=x_ret.index, columns=x_ret.columns, dtype=float)
    for s in x_ret.columns:
        out[s] = x_ret[s].rolling(win, min_periods=minp).corr(m_ret)
    return out.to_numpy(dtype=float)


# macro series reindexed onto panel grid (weekend NaN is fine)
macro = panel["macro"].reindex(idx)
print("macro coverage on panel:",
      {k: int(macro[k].notna().sum()) for k in macro.columns})
mret = {k: np.log(macro[k] / macro[k].shift(1)) for k in macro.columns}
mz = {k: (macro[k] - macro[k].rolling(120).mean()) / (macro[k].rolling(120).std() + 1e-12)
      for k in macro.columns}

cands = {}

# ============ A. macro sensitivity (beta/corr, per-symbol z) ============
for mk in ["DXY", "USDJPY", "EURUSD", "USDCNY", "VIX"]:
    r = mret[mk]
    cands[f"beta_{mk.lower()}_60"] = zscore_win(roll_beta(LRET, r), 250, 60)
    cands[f"corr_{mk.lower()}_60"] = zscore_win(roll_corr(LRET, r), 250, 60)

# ============ B. volume signals ============
logvol = np.log(V.clip(lower=1e-9))
cands["vol_z_60"] = ((logvol - logvol.rolling(60).mean()) / (logvol.rolling(60).std() + 1e-12)).to_numpy(dtype=float)
cands["vol_trend_20_60"] = (logvol.rolling(20).mean() / (logvol.rolling(60).mean() + 1e-12)).to_numpy(dtype=float)
amihud = (RET.abs() / (V + 1e-9)).clip(upper=1.0)
cands["amihud_20_z"] = zscore_win(np.log(amihud.clip(lower=1e-12)).rolling(20).mean().to_numpy(dtype=float), 250, 60)
cands["vol_of_vol_60"] = zscore_win(logvol.rolling(20).std().to_numpy(dtype=float), 250, 60)
volcv_20 = logvol.rolling(20).std() / (logvol.rolling(20).mean() + 1e-12)
cands["vol_cv_20_z"] = zscore_win(volcv_20.to_numpy(dtype=float), 250, 60)

# ============ C. yield-curve sensitivity ============
# US10Y/CN10Y are in the asset universe itself; use cross-asset bond beta from universe
cands["beta_us10y_60"] = zscore_win(roll_beta(LRET, LRET["US10Y"]), 250, 60)
cands["beta_cn10y_60"] = zscore_win(roll_beta(LRET, LRET["CN10Y"]), 250, 60)
cands["bond_beta_diff_60"] = zscore_win(
    (roll_beta(LRET, LRET["US10Y"]) - roll_beta(LRET, LRET["CN10Y"])), 250, 60)
cands["corr_us10y_cn10y_60"] = zscore_win(roll_corr(LRET, LRET["US10Y"]), 250, 60)

# ============ D. macro-conditioned intraday reversal (re-check rho) ============
intra_rev = (1.0 - C / O)  # == id_rev_1d sign
cands["rev_intra_z120"] = zscore_win(intra_rev.to_numpy(dtype=float), 120, 40)
vix_hi = (mz["VIX"] > 0.0).astype(float)
cands["rev_intra_x_vixhi"] = zscore_win(intra_rev.mul(vix_hi, axis=0).to_numpy(dtype=float), 120, 40)
dxy_up = (mz["DXY"] > 0.0).astype(float)
cands["rev_intra_x_dxyhi"] = zscore_win(intra_rev.mul(dxy_up, axis=0).to_numpy(dtype=float), 120, 40)
cands["rev1_x_vixhi"] = zscore_win((-LRET).mul(vix_hi, axis=0).to_numpy(dtype=float), 120, 40)

# ============ E. cross-sectional deviation / dispersion ============
ret20 = C.pct_change(20)
xs_mean = ret20.mean(axis=1)
xs_std = ret20.std(axis=1)
cands["xs_dev_20"] = ((ret20.sub(xs_mean, axis=0)).div(xs_std + 1e-12, axis=0)).to_numpy(dtype=float)
xs_mean5 = C.pct_change(5).mean(axis=1)
cands["xs_dev_5"] = (C.pct_change(5).sub(xs_mean5, axis=0)).to_numpy(dtype=float)
xs_rank_rng20 = (H.rolling(20).max() - L.rolling(20).min()).rank(axis=1, pct=True)
cands["xs_rank_range_20"] = xs_rank_rng20.to_numpy(dtype=float)

# ============ F. macro-momentum interaction on own return ============
# asset's own 1d return interacted with dollar regime (risk-on/off)
dxy_lo = (mz["DXY"] < 0.0).astype(float)
cands["ret1_x_dxylo"] = zscore_win(LRET.mul(dxy_lo, axis=0).to_numpy(dtype=float), 120, 40)
jpy_up = (mz["USDJPY"] > 0.0).astype(float)
cands["ret1_x_jpyup"] = zscore_win(LRET.mul(jpy_up, axis=0).to_numpy(dtype=float), 120, 40)


# ---------------- IC machinery ----------------
def ic_stats(fac_np, fwd=fwd1):
    df = pd.DataFrame(fac_np, index=idx, columns=SYMS)
    ev = df.loc[ev_idx]
    fwd_ev = fwd.loc[ev_idx]
    ics = []
    for dt in ev.index:
        f = ev.loc[dt].dropna()
        r = fwd_ev.loc[dt].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < 8:
            continue
        x = f[common].astype(float).rank()
        y = r[common].astype(float).rank()
        if x.std() == 0 or y.std() == 0:
            continue
        ic = np.corrcoef(x, y)[0, 1]
        if np.isfinite(ic):
            ics.append(ic)
    ics = np.array(ics)
    if len(ics) == 0:
        return None
    sd = ics.std(ddof=1)
    return dict(ic=float(ics.mean()),
                icir=float(ics.mean() / sd) if sd > 1e-12 else 0.0,
                hit=float((ics > 0).mean()), n=int(len(ics)), ics=ics)


rows = []
for name, fac in cands.items():
    st = ic_stats(fac)
    if st is None:
        print(f"{name:28s} NO VALID IC")
        continue
    df = pd.DataFrame(fac, index=idx, columns=SYMS)
    ev = df.loc[ev_idx]
    cov = float(ev.notna().mean().mean())
    rk = df.rank(axis=1, pct=True)
    turn = float((rk.loc[ev_idx] - rk.loc[ev_idx].shift(10)).abs().mean().mean())
    rhos = [abs(pooled_rho(fac, v)) for v in lib_np.values()]
    rho_max = max(rhos) if rhos else float("nan")
    rho_anchor = abs(pooled_rho(fac, lib_np["mom_10d_skip5"]))
    qual = abs(st["ic"]) * abs(st["icir"])
    flag = "PASS" if (abs(st["ic"]) >= GATE_IC and abs(st["icir"]) >= GATE_ICIR and rho_max < 0.5) else "    "
    rows.append((name, st["ic"], st["icir"], st["hit"], st["n"], cov, turn, rho_max, rho_anchor, qual))
    print(f"{flag} {name:28s} IC1={st['ic']:+.4f} ICIR1={st['icir']:+.3f} hit={st['hit']:.3f} "
          f"n={st['n']} cov={cov:.3f} turn={turn:.3f} rho_max={rho_max:+.3f} rho_anchor={rho_anchor:+.3f} qual={qual:.4f}")

print("\n--- passing IC gates sorted by quality ---")
for r in sorted(rows, key=lambda x: -x[9]):
    name, ic, icir, hit, n, cov, turn, rho_max, rho_anchor, qual = r
    if abs(ic) >= GATE_IC and abs(icir) >= GATE_ICIR:
        print(f"{name:28s} IC1={ic:+.4f} ICIR1={icir:+.3f} hit={hit:.3f} n={n} cov={cov:.3f} "
              f"turn={turn:.3f} rho_max={rho_max:+.3f} rho_anchor={rho_anchor:+.3f} qual={qual:.4f} "
              f"{'DIVERSE' if rho_max < 0.5 else 'CORRELATED'}")

out = {r[0]: dict(ic=r[1], icir=r[2], hit=r[3], n=r[4], cov=r[5], turn=r[6],
                  rho_max=r[7], rho_anchor=r[8], qual=r[9]) for r in rows}
with open("scripts/miner3_screen_cycle8_results.json", "w") as fh:
    json.dump(out, fh, indent=1)
print("\nsaved scripts/miner3_screen_cycle8_results.json")
