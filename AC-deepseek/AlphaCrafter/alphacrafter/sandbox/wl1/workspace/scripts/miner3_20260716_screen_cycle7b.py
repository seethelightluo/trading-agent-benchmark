"""miner_3 cycle-7 screening: orthogonal factor families on FULL 2388-row panel.

Context:
- Library (14 distinct effective factors) is dominated by 1-5d mean-reversion
  signals (rev_1d/2d/3d, nclv_1d..5d, id_rev, nbody, cz_rev1, rev1_pk,
  rev1_x_inveff, rev_1d_vs) plus mom_10d_skip5 (used negatively = reversal).
- Previous miner3 candidates got evicted by the post-Miner gate because pooled
  rho vs miner2_20260716_mom_10d_skip5 exceeded 0.5 (often ~0.88 due to
  shared short-horizon return structure).
- This cycle tests genuinely different families: long-horizon reversal,
  volatility asymmetry/tail, trend structure, cross-asset beta (BTC/US10Y/XAU/
  WTI), overnight/intraday share, volume-conditioned, and robust RSI-2.

Gate: |IC1| >= 0.007, |ICIR1| >= 0.084 (1d forward rank IC on
2021-01-04..2026-07-15), max_abs_library_correlation < 0.5 (pooled spearman
on the full panel grid).
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
assert idx.is_unique, "duplicate dates in panel"
LRET = np.log(C / C.shift(1))
RET = C.pct_change()
fwd1 = RET.shift(-1)
ev_idx = idx[(idx >= EVAL_START) & (idx <= END)]

# ---------------- reconstruct all 14 library signals on the full panel ----------------
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

# verify against real artifact
mom_art = np.load("factors/miner2_20260716_mom_10d_skip5.npy", allow_pickle=True)
a = lib_np["mom_10d_skip5"].ravel(); b = mom_art.astype(float).ravel()
m = (~np.isnan(a)) & (~np.isnan(b))
print("lib recon check mom_10d_skip5 pooled rho vs .npy: %.8f (%d pairs)" % (spearmanr(a[m], b[m]).statistic, m.sum()))


def pooled_rho(a, b, min_n=500):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    m = (~np.isnan(a)) & (~np.isnan(b))
    if m.sum() < min_n:
        return float("nan")
    return float(spearmanr(a[m], b[m]).statistic)


def zscore_win(X, win=120, minp=40):
    df = pd.DataFrame(X, index=idx, columns=SYMS)
    mu = df.rolling(win, min_periods=minp).mean()
    sd = df.rolling(win, min_periods=minp).std()
    return ((df - mu) / (sd + 1e-12)).to_numpy(dtype=float)


def zscore_full(X):
    """per-symbol full-sample z-score (removes cross-asset level effects)"""
    df = pd.DataFrame(X, index=idx, columns=SYMS)
    return df.sub(df.mean()).div(df.std() + 1e-12).to_numpy(dtype=float)


# ---------------- candidates ----------------
cands = {}

# A. long-horizon reversal (classic, distinct from 1-5d library)
cands["rev_120d_z"] = zscore_win(-(np.log(C) - np.log(C.shift(120))).to_numpy(dtype=float), 250, 60)
cands["rev_250d_z"] = zscore_win(-(np.log(C) - np.log(C.shift(250))).to_numpy(dtype=float), 500, 120)

# B. volatility asymmetry / tail
pos = RET.where(RET > 0, 0.0)
neg = (-RET).where(RET < 0, 0.0)
cands["down_up_vol_20"] = (neg.rolling(20).std() / (pos.rolling(20).std() + 1e-12)).to_numpy(dtype=float)
std20 = LRET.rolling(20).std()
tail = (RET.abs() > 2 * std20).astype(float)
cands["tail_freq_60"] = tail.rolling(60).mean().to_numpy(dtype=float)
cands["skew_60_z"] = zscore_win(LRET.rolling(60).skew().to_numpy(dtype=float), 250, 60)
cands["kurt_60_z"] = zscore_win(LRET.rolling(60).kurt().to_numpy(dtype=float), 250, 60)
rng = (H - L) / C.shift(1)
rng_mean = rng.rolling(20).mean()
rng_sd = rng.rolling(20).std()
cands["range_shock_20"] = ((rng - rng_mean) / (rng_sd + 1e-12)).to_numpy(dtype=float)

# C. trend structure
cands["eff_ratio_20"] = ((C - C.shift(20)).abs() / (C.diff().abs().rolling(20).sum() + 1e-12)).to_numpy(dtype=float)
sign = np.sign(RET)
cands["streak_10"] = sign.rolling(10).sum().to_numpy(dtype=float)
cands["autocorr1_60"] = LRET.rolling(60).apply(lambda x: pd.Series(x).autocorr(1) if len(x) > 10 else np.nan, raw=True).to_numpy(dtype=float)
cands["ts_mom_60_z"] = zscore_win((np.log(C) - np.log(C.shift(60))).to_numpy(dtype=float), 250, 60)
cands["ts_mom_120_z"] = zscore_win((np.log(C) - np.log(C.shift(120))).to_numpy(dtype=float), 500, 120)

# D. cross-asset beta / correlation (new family)
def roll_corr(x, y, win=60):
    return x.rolling(win).corr(y)

cands["corr_btc_60"] = zscore_full(roll_corr(LRET, LRET["BTC"], 60).to_numpy(dtype=float))
cands["corr_us10y_60"] = zscore_full(roll_corr(LRET, LRET["US10Y"], 60).to_numpy(dtype=float))
cands["corr_xau_60"] = zscore_full(roll_corr(LRET, LRET["XAU"], 60).to_numpy(dtype=float))
cands["corr_wti_60"] = zscore_full(roll_corr(LRET, LRET["WTI"], 60).to_numpy(dtype=float))
btc_ret = LRET["BTC"]
_beta = pd.DataFrame(index=idx, columns=SYMS, dtype=float)
for _s in SYMS:
    _beta[_s] = LRET[_s].rolling(60).cov(btc_ret) / (btc_ret.rolling(60).var() + 1e-12)
cands["beta_btc_60"] = zscore_full(_beta.to_numpy(dtype=float))

# E. overnight / intraday share
gaps = O / C.shift(1) - 1.0
intra = C / O - 1.0
cands["on_share_20"] = (gaps.abs().rolling(20).sum() / (gaps.abs().rolling(20).sum() + intra.abs().rolling(20).sum() + 1e-12)).to_numpy(dtype=float)
cands["gap_cum_5_z"] = zscore_win(gaps.rolling(5).sum().to_numpy(dtype=float), 120, 40)
cands["gap_cum_20_z"] = zscore_win(gaps.rolling(20).sum().to_numpy(dtype=float), 250, 60)

# F. robust RSI (2d and 5d) with coverage diagnostics
def rsi_robust(win):
    up = RET.clip(lower=0).rolling(win).mean()
    dn = (-RET.clip(upper=0)).rolling(win).mean()
    rs = up / (dn + 1e-9)
    r = 100 - 100 / (1 + rs)
    return r

cands["rsi_2"] = rsi_robust(2).to_numpy(dtype=float)
cands["rsi_5"] = rsi_robust(5).to_numpy(dtype=float)

# G. 5d/10d return reversal, per-symbol z-scored (longer than library 1-3d)
cands["rev_5d_z"] = zscore_win(-(np.log(C) - np.log(C.shift(5))).to_numpy(dtype=float), 120, 40)
cands["rev_10d_z"] = zscore_win(-(np.log(C) - np.log(C.shift(10))).to_numpy(dtype=float), 250, 60)

# H. momentum variants (skip recent to reduce correlation with mom_10d_skip5)
cands["mom_20_skip5_z"] = zscore_win((np.log(C.shift(5)) - np.log(C.shift(20))).to_numpy(dtype=float), 250, 60)
cands["mom_60_skip20_z"] = zscore_win((np.log(C.shift(20)) - np.log(C.shift(60))).to_numpy(dtype=float), 500, 120)

# I. volume-conditioned reversal (coverage ~9/15; z-scored to decouple)
vol_ok = V > 0
vol_z = zscore_win(V.to_numpy(dtype=float), 120, 40)
cands["vol_cond_rev1"] = (-LRET * (vol_z > 0.5).astype(float)).to_numpy(dtype=float)

# J. macro-conditioned intraday reversal (VIX high regime), z-scored
vix = panel["macro"]["VIX"]
vix_z = (vix - vix.rolling(120).mean()) / (vix.rolling(120).std() + 1e-12)
intra_rev = (1.0 - C / O)
cands["rev_intra_x_vixhi"] = zscore_win((intra_rev * (vix_z > 0.0).astype(float)).to_numpy(dtype=float), 120, 40)

# K. vol level z (60d) - previously passed gates but rho high; try per-symbol z
cands["vol_z_60"] = zscore_win(LRET.rolling(60).std().to_numpy(dtype=float), 250, 60)


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
        print(f"{name:20s} NO VALID IC")
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
    print(f"{flag} {name:20s} IC1={st['ic']:+.4f} ICIR1={st['icir']:+.3f} hit={st['hit']:.3f} "
          f"n={st['n']} cov={cov:.3f} turn={turn:.3f} rho_max={rho_max:+.3f} rho_anchor={rho_anchor:+.3f} qual={qual:.4f}")

print("\n--- candidates passing IC gates (sorted by quality) ---")
for r in sorted(rows, key=lambda x: -x[9]):
    name, ic, icir, hit, n, cov, turn, rho_max, rho_anchor, qual = r
    if abs(ic) >= GATE_IC and abs(icir) >= GATE_ICIR:
        print(f"{name:20s} IC1={ic:+.4f} ICIR1={icir:+.3f} hit={hit:.3f} n={n} cov={cov:.3f} "
              f"turn={turn:.3f} rho_max={rho_max:+.3f} rho_anchor={rho_anchor:+.3f} qual={qual:.4f} "
              f"{'DIVERSE' if rho_max < 0.5 else 'CORRELATED'}")

out = {r[0]: dict(ic=r[1], icir=r[2], hit=r[3], n=r[4], cov=r[5], turn=r[6],
                  rho_max=r[7], rho_anchor=r[8], qual=r[9]) for r in rows}
with open("scripts/miner3_screen_cycle7_results.json", "w") as fh:
    json.dump(out, fh, indent=1)
print("\nsaved scripts/miner3_screen_cycle7_results.json")
