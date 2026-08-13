"""Regime + factor recent-IC check using ONLY data visible through 2033-03-18 (fixed)."""
import os
import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp("2033-03-18")
DATA = "../persistent/stock_data"
IDX = "../persistent/index_data"
ASSETS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU",
          "COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
OBS = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]

def load_close(sym, d=DATA):
    df = pd.read_csv(os.path.join(d, sym + ".csv"), parse_dates=["date"]).set_index("date")
    return df["close"].astype(float)

closes = pd.DataFrame({s: load_close(s) for s in ASSETS}).loc[:CUTOFF]
obs = pd.DataFrame({s: load_close(s, d=IDX) for s in OBS}).loc[:CUTOFF]
rets = closes.pct_change()
obs_rets = obs.pct_change()

H = 10
fwd = closes.shift(-H) / closes - 1
mkt = closes.mean(axis=1)
mkt_ret = mkt.pct_change()

# factor 1: vol_adj_mom_accel_20x60
mom20 = closes / closes.shift(20) - 1
mom60 = closes / closes.shift(60) - 1
vol20 = rets.rolling(20).std()
f_mom = (mom20 - mom60) / vol20

# factor 2: dn_mkt_beta_60d (beta on down-market days)
down = mkt_ret.clip(upper=0)
f_beta = pd.DataFrame(index=closes.index, columns=ASSETS, dtype=float)
for s in ASSETS:
    f_beta[s] = rets[s].rolling(60).cov(down) / down.rolling(60).var()

# factor 3: rate_beta_cn10y_60d (beta on CN10Y yield change)
cn10y_chg = closes["CN10Y"].pct_change()
f_rate = pd.DataFrame(index=closes.index, columns=ASSETS, dtype=float)
for s in ASSETS:
    f_rate[s] = rets[s].rolling(60).cov(cn10y_chg) / cn10y_chg.rolling(60).var()

def recent_ic(fac, name, direction):
    pairs = []
    for t in fac.index:
        if t not in fwd.index:
            continue
        srow, frow = fac.loc[t], fwd.loc[t]
        m = srow.notna() & frow.notna() & np.isfinite(srow.astype(float)) & np.isfinite(frow.astype(float))
        if m.sum() >= 8:
            pairs.append((t, np.corrcoef(srow[m].astype(float), frow[m].astype(float))[0, 1]))
    if not pairs:
        print(f"{name}: no dates"); return pd.Series(dtype=float)
    idx, vals = zip(*pairs)
    ic = pd.Series(vals, index=pd.DatetimeIndex(idx))
    out = {}
    for w in [20, 60, 120, 250]:
        sub = ic.tail(w)
        if len(sub) >= 10:
            out[w] = (sub.mean(), sub.mean()/sub.std(), (sub>0).mean(), len(sub))
    print(f"{name} (dir {direction:+d}): " + " | ".join(
        f"IC{w}d={v[0]:+.3f} ICIR={v[1]:+.3f} hit={v[2]:.0%} n={v[3]}" for w, v in out.items()))
    return ic

ic_mom = recent_ic(f_mom, "vol_adj_mom_accel_20x60", 1)
ic_beta = recent_ic(f_beta, "dn_mkt_beta_60d", 1)
ic_rate = recent_ic(f_rate, "rate_beta_cn10y_60d", -1)

print("\n=== FACTOR PAIRWISE CS-CORR (last 120d avg) ===")
last120 = closes.index[-120:]
for a, b in [("mom","beta"),("mom","rate"),("beta","rate")]:
    fa, fb = {"mom": f_mom, "beta": f_beta, "rate": f_rate}[a], {"mom": f_mom, "beta": f_beta, "rate": f_rate}[b]
    cs_corrs = []
    for t in last120:
        x, y = fa.loc[t].astype(float), fb.loc[t].astype(float)
        m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 8:
            cs_corrs.append(np.corrcoef(x[m], y[m])[0, 1])
    print(f"{a}-{b}: mean cs-corr {np.nanmean(cs_corrs):+.3f} (n={len(cs_corrs)})")

print("\n=== LATEST CROSS-SECTIONAL SIGNAL RANKS (2033-03-18) ===")
for name, fac, direction in [("vol_adj_mom_accel_20x60", f_mom, 1),
                              ("dn_mkt_beta_60d", f_beta, 1),
                              ("rate_beta_cn10y_60d", f_rate, -1)]:
    row = fac.iloc[-1].astype(float)
    eff = row * direction
    ranked = eff.dropna().rank(ascending=False)
    print(f"{name} (dir{direction:+d}): " + ", ".join(f"{s}={eff[s]:+.3f}(r{ranked[s]:.0f})" for s in ASSETS if np.isfinite(eff[s])))
