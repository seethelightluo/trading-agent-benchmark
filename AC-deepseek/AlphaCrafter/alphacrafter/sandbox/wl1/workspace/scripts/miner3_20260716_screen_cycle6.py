"""miner_3 cycle-6 screening on FULL 2388-row panel with gate-compatible artifacts.

Learnings from cycle-19 audit:
- The post-Miner gate loads artifacts on the full 2388x15 panel grid (2020-01-01..
  2026-07-15, crypto calendar) and computes pooled pairwise spearman rho.
- My earlier eval-window (1172-row) embedded artifacts were misaligned vs the
  library .npy (2388 rows) -> spurious rho ~0.89 -> evicted.
- quality = |IC1| * |ICIR1| (matches gate's reported values exactly).

Here every candidate is built on the full panel, metrics on 2021-01-04..2026-07-15,
and pooled rho is computed vs miner2_20260716_mom_10d_skip5.npy with correct
alignment. Gate: |IC1|>=0.007, |ICIR1|>=0.084, rho<0.5.
"""
import sys, os, json
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
LRET = np.log(C / C.shift(1))
RET = C.pct_change()
fwd1 = RET.shift(-1)
idx = C.index
ev_idx = idx[(idx >= EVAL_START) & (idx <= END)]

mom_lib = np.load("factors/miner2_20260716_mom_10d_skip5.npy", allow_pickle=True)
print("mom npy:", mom_lib.shape, "| full panel:", C.shape)


def pooled_rho(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = (~np.isnan(a)) & (~np.isnan(b))
    if m.sum() < 500:
        return float("nan")
    return float(spearmanr(a[m], b[m]).statistic)


def zscore_cols(X):
    Z = np.full_like(X, np.nan, dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        mu = np.nanmean(col)
        sd = np.nanstd(col)
        if sd > 0:
            Z[:, j] = (col - mu) / sd
    return Z


def zscore_cols_win(X, win=120):
    """per-symbol rolling z-score over trailing window"""
    df = pd.DataFrame(X, index=idx, columns=SYMS)
    mu = df.rolling(win, min_periods=40).mean()
    sd = df.rolling(win, min_periods=40).std()
    return ((df - mu) / (sd + 1e-12)).to_numpy(dtype=float)


# ---------------- candidate library ----------------
cands = {}

# A. VWAP deviation 20d (liquidity-weighted trend position)
vwap20 = (C * V).rolling(20).sum() / V.rolling(20).sum()
cands["vwap_dev_20"] = (C / vwap20 - 1.0).to_numpy(dtype=float)
cands["vwap_slope_20"] = (vwap20 / vwap20.shift(20) - 1.0).to_numpy(dtype=float)

# B. volatility term structure
std5 = LRET.rolling(5).std()
std10 = LRET.rolling(10).std()
std20 = LRET.rolling(20).std()
std60 = LRET.rolling(60).std()
cands["vol_term_5_20"] = np.log(std5 / std20).to_numpy(dtype=float)
cands["vol_term_10_60"] = np.log(std10 / std60).to_numpy(dtype=float)

# C. extreme-return reversal, short horizon
cands["max_ret_5"] = RET.rolling(5).max().to_numpy(dtype=float)
cands["min_ret_5"] = RET.rolling(5).min().to_numpy(dtype=float)

# D. CCI-20 (commodity channel index)
md = (C - C.rolling(20).mean()).abs().rolling(20).mean()
cands["cci_20"] = ((C - C.rolling(20).mean()) / (0.015 * md + 1e-12)).to_numpy(dtype=float)

# E. RSI-2 (short-term mean reversion)
def rsi(win):
    up = RET.clip(lower=0).rolling(win).mean()
    dn = (-RET.clip(upper=0)).rolling(win).mean()
    rs = up / (dn + 1e-12)
    return 100 - 100 / (1 + rs)

cands["rsi_2"] = rsi(2).to_numpy(dtype=float)

# F. trend consistency breadth 60d: fraction of days close>SMA20
sma20 = C.rolling(20).mean()
cands["trend_consist_60"] = (C > sma20).rolling(60).mean().to_numpy(dtype=float)

# G. vol-normalized SMA slope
cands["slope_norm_20"] = ((sma20 / sma20.shift(10) - 1.0) / (std20 + 1e-12)).to_numpy(dtype=float)

# H. rolling-z intraday reversal (standardized, to decouple from level factors)
cands["rev_intra_z120"] = zscore_cols_win((1.0 - C / O).to_numpy(dtype=float), 120)

# I. 5d cumulative gap z-scored (overnight momentum)
gaps = (O / C.shift(1) - 1.0)
cands["gap_mom_5_z"] = zscore_cols_win(gaps.rolling(5).sum().to_numpy(dtype=float), 120)

# J. momentum 60d skip10, per-symbol z-scored
mom60 = (C / C.shift(60) - 1.0) * (1 - (C.shift(10) / C.shift(60) - 1.0))  # placeholder, replace below
mom60 = (C.shift(10) / C.shift(60) - 1.0)
cands["mom_60_z"] = zscore_cols_win(mom60.to_numpy(dtype=float), 250)

# K. rolling corr with NDX (equity-breadth beta), 60d
cands["corr_ndx_60"] = LRET.rolling(60).corr(LRET["NDX"]).to_numpy(dtype=float)

# L. distance from 60d high (drawdown position)
cands["dist_high_60"] = (C / C.rolling(60).max() - 1.0).to_numpy(dtype=float)

# M. Parkinson vol (intraday range-based) z-scored - level factor but z-scored
pv = np.sqrt(np.log(H / L) ** 2 / (4 * np.log(2)))
cands["park_vol_z"] = zscore_cols_win(pv.to_numpy(dtype=float), 120)

# N. skewness 20d (roll)
cands["skew_20_z"] = zscore_cols_win(LRET.rolling(20).skew().to_numpy(dtype=float), 250)


def ic_stats(fac_np, fwd=fwd1, rank=True):
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
        x = f[common].astype(float)
        y = r[common].astype(float)
        if rank:
            x = x.rank()
            y = y.rank()
        if x.std() == 0 or y.std() == 0:
            continue
        ic = np.corrcoef(x, y)[0, 1]
        if np.isfinite(ic):
            ics.append(ic)
    ics = np.array(ics)
    if len(ics) == 0:
        return None
    return dict(ic=float(ics.mean()),
                icir=float(ics.mean() / ics.std(ddof=1)) if ics.std(ddof=1) > 1e-12 else 0.0,
                hit=float((ics > 0).mean()), n=int(len(ics)))


rows = []
for name, fac in cands.items():
    st = ic_stats(fac)
    if st is None:
        print(f"{name:18s} no valid IC")
        continue
    df = pd.DataFrame(fac, index=idx, columns=SYMS)
    ev = df.loc[ev_idx]
    cov = float(ev.notna().mean().mean())
    rk = df.rank(axis=1, pct=True)
    turn = float((rk.loc[ev_idx] - rk.loc[ev_idx].shift(10)).abs().mean().mean())
    rho_mom = pooled_rho(fac, mom_lib)
    qual = abs(st["ic"]) * abs(st["icir"])
    flag = "PASS" if (abs(st["ic"]) >= GATE_IC and abs(st["icir"]) >= GATE_ICIR and abs(rho_mom) < 0.5) else "    "
    rows.append((name, st["ic"], st["icir"], st["hit"], st["n"], cov, turn, rho_mom, qual))
    print(f"{flag} {name:18s} IC1={st['ic']:+.4f} ICIR1={st['icir']:+.3f} hit={st['hit']:.3f} "
          f"n={st['n']} cov={cov:.3f} turn={turn:.3f} rho_mom={rho_mom:+.3f} qual={qual:.4f}")

print("\n--- passing gates (incl rho<0.5) ---")
for r in sorted(rows, key=lambda x: -x[8]):
    name, ic, icir, hit, n, cov, turn, rho_mom, qual = r
    if abs(ic) >= GATE_IC and abs(icir) >= GATE_ICIR:
        print(f"{name:18s} IC1={ic:+.4f} ICIR1={icir:+.3f} hit={hit:.3f} n={n} cov={cov:.3f} "
              f"turn={turn:.3f} rho_mom={rho_mom:+.3f} qual={qual:.4f} {'DIVERSE' if abs(rho_mom) < 0.5 else 'CORRELATED'}")

with open("scripts/miner3_screen_cycle6_results.json", "w") as fh:
    json.dump({r[0]: dict(ic=r[1], icir=r[2], hit=r[3], n=r[4], cov=r[5], turn=r[6], rho_mom=r[7], qual=r[8])
               for r in rows}, fh, indent=1)
print("\nsaved scripts/miner3_screen_cycle6_results.json")
