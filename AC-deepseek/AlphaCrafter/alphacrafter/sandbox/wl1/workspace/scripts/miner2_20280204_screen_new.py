"""miner_2: screen new candidate factors through 2028-02-03.
Candidates: vol-scaled reversal, MA-zscore reversal, gap reversal, drawdown depth,
RSI-style, overnight/intraday split, vol-scaled nclv, cross-sectional vol-adjusted rev.
Gate: |IC|>=0.0070, |ICIR|>=0.0840 (1d horizon, full window 2021-01-01+).
Also report recent-400d IC and max |corr| vs existing library.
"""
import pandas as pd
import numpy as np
import pickle

panel = pickle.load(open("scripts/panel_cache.pkl", "rb"))
C, O, H, L, V = panel["close"], panel["open"], panel["high"], panel["low"], panel["vol"]
M = panel["macro"]
R = C.pct_change()
LOG = np.log(C)

def realized_vol(s, w):
    return s.pct_change().rolling(w).std()

# existing library (for correlation check)
LIB = {}
LIB["mom_120d_skip5"] = (C / C.shift(120).shift(5) - 1.0)
LIB["vol_of_vol20x60"] = realized_vol(C, 20) / realized_vol(C, 60)
LIB["nclv_1d"] = -(C - L) / (H - L)
LIB["nclv_2d"] = -(C - L.rolling(2).min()) / (H.rolling(2).max() - L.rolling(2).min())
LIB["nclv_3d"] = -(C - L.rolling(3).min()) / (H.rolling(3).max() - L.rolling(3).min())
LIB["rev_1d"] = -LOG.diff(1)
LIB["rev_2d"] = -(LOG - LOG.shift(2))
LIB["rev_3d"] = -(LOG - LOG.shift(3))
LIB["rev_5d"] = -(LOG - LOG.shift(5))
LIB["id_rev_1d"] = -(C / O - 1.0)
LIB["nbody_1d"] = -(C - O) / (H - L)

# ---- new candidates ----
NEW = {}
rv5 = realized_vol(C, 5)
rv10 = realized_vol(C, 10)
rv20 = realized_vol(C, 20)

# N1: volatility-scaled 1d reversal
NEW["volsc_rev_1d"] = -(LOG.diff(1)) / rv5
# N2: volatility-scaled 2d reversal
NEW["volsc_rev_2d"] = -(LOG - LOG.shift(2)) / rv10
# N3: MA-zscore mean reversion (10d)
ma10 = C.rolling(10).mean(); sd10 = C.rolling(10).std()
NEW["ma_z_rev_10d"] = -(C - ma10) / sd10
# N4: MA-zscore mean reversion (20d)
ma20 = C.rolling(20).mean(); sd20 = C.rolling(20).std()
NEW["ma_z_rev_20d"] = -(C - ma20) / sd20
# N5: drawdown depth 60d (oversold bounce)
NEW["dd_60d"] = -(C / C.rolling(60).max() - 1.0)
# N6: gap reversal (open vs prev close)
NEW["gap_rev_1d"] = -(O / C.shift(1) - 1.0)
# N7: intraday reversal (close vs open), scaled
NEW["intra_rev_1d"] = -(C / O - 1.0)
# N8: vol-scaled close location value 1d
NEW["volsc_nclv_1d"] = -(C - L) / (H - L) * (1.0 / (1.0 + rv20))
# N9: RSI-14 style reversal (normalized)
def rsi_style(s, w=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(w).mean()
    dn = (-d.clip(upper=0)).rolling(w).mean()
    rs = up / (dn + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)
NEW["rsi_rev_14d"] = -(rsi_style(C) - 50.0)
# N10: 5d reversal scaled by volume z
vz = V / V.rolling(20).mean()
NEW["rev_5d_vz"] = -(LOG - LOG.shift(5)) * vz
# N11: efficiency-ratio reversal (trend quality contrarian)
eff = (C - C.shift(10)).abs() / (C.diff().abs().rolling(10).sum())
NEW["eff_rev_10d"] = -eff
# N12: overnight vs intraday spread (mean reversion of close-to-close decompositions)
NEW["oi_spread_1d"] = -((O / C.shift(1) - 1.0) - (C / O - 1.0))

def rank_ic_series(F, fwd, min_valid=8):
    rF = F.rank(axis=1)
    rR = fwd.rank(axis=1)
    mask = F.notna() & fwd.notna()
    n = mask.sum(axis=1)
    def zscore(df, m):
        out = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
        out[m] = df[m]
        mu = out.mean(axis=1)
        sd = out.std(axis=1)
        return (out.sub(mu, axis=0)).div(sd, axis=0)
    zF = zscore(rF, mask)
    zR = zscore(rR, mask)
    ic = (zF * zR).sum(axis=1) / (n - 1).clip(lower=1)
    ic = ic.where(n >= min_valid)
    return ic.dropna()

def fwd_ret(h):
    return C.shift(-h) / C - 1.0

FULL0 = pd.Timestamp("2021-01-01")

def metrics(s):
    if len(s) < 50:
        return np.nan, np.nan, np.nan, len(s)
    return s.mean(), (s.mean() / s.std() if s.std() > 0 else 0.0), (s > 0).mean(), len(s)

lib_flat = {k: v.stack().dropna() for k, v in LIB.items()}
print(f"{'factor':<16} {'IC1':>7} {'ICIR1':>7} {'hit':>6} {'N':>5} | {'recIC':>7} {'recICIR':>7} | {'maxLib':>6} | {'cov':>5}")
for name, F in NEW.items():
    Ff = F.loc[F.index >= FULL0]
    s = rank_ic_series(Ff, fwd_ret(1).loc[Ff.index])
    ic, icir, hit, n = metrics(s)
    rec = s[s.index >= s.index[-1] - pd.Timedelta(days=400)]
    ric, ricir, rhit, rn = metrics(rec)
    fflat = F.stack().dropna()
    maxc = 0.0
    for k, lf in lib_flat.items():
        j = fflat.index.intersection(lf.index)
        if len(j) > 200:
            c = np.corrcoef(fflat.loc[j].values, lf.loc[j].values)[0, 1]
            maxc = max(maxc, abs(c))
    cov = Ff.notna().mean().mean()
    flag = "  <-- PASS" if (abs(ic) >= 0.007 and abs(icir) >= 0.084) else ""
    print(f"{name:<16} {ic:>7.4f} {icir:>7.3f} {hit:>6.3f} {n:>5d} | {ric:>7.4f} {ricir:>7.3f} | {maxc:>6.3f} | {cov:>5.2f}{flag}")
