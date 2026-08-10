"""miner_3 cycle-6 diversity/pair checks: RSI-family and intraday-reversal z variants.
Tests IC/ICIR vs admission gates and pairwise rho vs library artifacts (mom, er20)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr

panel = pd.read_pickle("scripts/panel_cache.pkl")
C = panel["close"].astype(float)
O = panel["open"].astype(float)
SYMS = list(C.columns)
RET = C.pct_change()
fwd1 = RET.shift(-1)
idx = C.index
EVAL_START = pd.Timestamp("2021-01-04")
EVAL_END = pd.Timestamp("2026-07-15")
ev_idx = idx[(idx >= EVAL_START) & (idx <= EVAL_END)]
mom_lib = np.load("factors/miner2_20260716_mom_10d_skip5.npy")
er_lib = np.load("factors/miner1_20260716_er20.npy")


def pooled_rho(a, b, min_n=300):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = (~np.isnan(a)) & (~np.isnan(b))
    if m.sum() < min_n:
        return float("nan")
    return float(spearmanr(a[m], b[m]).statistic)


def daily_mean_rho(a, b, min_dates=50):
    """mean over dates of cross-sectional spearman rho (robustness check)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    rhos = []
    for i in range(a.shape[0]):
        x, y = a[i], b[i]
        m = (~np.isnan(x)) & (~np.isnan(y))
        if m.sum() < 8:
            continue
        xr, yr = x[m], y[m]
        if xr.std() == 0 or yr.std() == 0:
            continue
        r = spearmanr(xr, yr).statistic
        if np.isfinite(r):
            rhos.append(r)
    rhos = np.array(rhos)
    return float(rhos.mean()) if len(rhos) >= min_dates else float("nan")


def zscore_cols_win(X, win):
    df = pd.DataFrame(X, index=idx, columns=SYMS)
    mu = df.rolling(win, min_periods=40).mean()
    sd = df.rolling(win, min_periods=40).std()
    return ((df - mu) / (sd + 1e-12)).to_numpy(dtype=float)


def rsi(win):
    up = RET.clip(lower=0).rolling(win).mean()
    dn = (-RET.clip(upper=0)).rolling(win).mean()
    rs = up / (dn + 1e-12)
    return 100 - 100 / (1 + rs)


def ic_stats(fac_np):
    df = pd.DataFrame(fac_np, index=idx, columns=SYMS)
    ev, fwd_ev = df.loc[ev_idx], fwd1.loc[ev_idx]
    ics, cov = [], []
    for dt in ev.index:
        f = ev.loc[dt].dropna()
        r = fwd_ev.loc[dt].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < 8:
            continue
        x, y = f[common].astype(float).rank(), r[common].astype(float).rank()
        if x.std() == 0 or y.std() == 0:
            continue
        ic = np.corrcoef(x, y)[0, 1]
        if np.isfinite(ic):
            ics.append(ic)
            cov.append(len(common))
    ics = np.array(ics)
    if len(ics) == 0:
        return None
    return dict(ic=float(ics.mean()),
                icir=float(ics.mean() / ics.std(ddof=1)) if ics.std(ddof=1) > 1e-12 else 0.0,
                hit=float((ics > 0).mean()), n=int(len(ics)),
                cov=float(np.mean(cov)))


def turnover(fac_np):
    df = pd.DataFrame(fac_np, index=idx, columns=SYMS)
    ev = df.loc[ev_idx]
    rk = ev.rank(axis=1)
    chg = rk.diff().abs().mean(axis=1).dropna()
    return float(chg.mean()) if len(chg) else float("nan")


rev = (1.0 - C / O).to_numpy(dtype=float)
cands = {
    "rsi_2": rsi(2).to_numpy(dtype=float),
    "rsi_3": rsi(3).to_numpy(dtype=float),
    "rsi_5": rsi(5).to_numpy(dtype=float),
    "rev_intra_z60": zscore_cols_win(rev, 60),
    "rev_intra_z120": zscore_cols_win(rev, 120),
    "rev_intra_z250": zscore_cols_win(rev, 250),
}
rows = {}
for name, fac in cands.items():
    st = ic_stats(fac)
    if st is None:
        print(f"{name:16s} no valid IC stats"); continue
    rho_mom = pooled_rho(fac, mom_lib)
    rho_er = pooled_rho(fac, er_lib)
    d_rho_mom = daily_mean_rho(fac, mom_lib)
    qual = abs(st["ic"]) * abs(st["icir"])
    tov = turnover(fac)
    rows[name] = (st, rho_mom, rho_er, qual)
    print(f"{name:16s} IC={st['ic']:+.4f} ICIR={st['icir']:+.3f} hit={st['hit']:.3f} n={st['n']} "
          f"cov={st['cov']:.1f} tov={tov:.2f} | rho_mom={rho_mom:+.3f} d_rho_mom={d_rho_mom:+.3f} "
          f"rho_er={rho_er:+.3f} qual={abs(st['ic'])*abs(st['icir']):.4f}")

print("\npairwise rho among candidates (pooled):")
names = list(rows)
for i in range(len(names)):
    for j in range(i+1, len(names)):
        r = pooled_rho(cands[names[i]], cands[names[j]])
        print(f"  {names[i]:16s} vs {names[j]:16s}: {r:+.3f}")
