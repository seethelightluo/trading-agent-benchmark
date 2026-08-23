"""miner_3 factor candidate A: hi_lo_pos_40d
Trend-positioning factor: closeness of close to the 40-day high-low range top.
Construction: (close - min(low,40)) / (max(high,40) - min(low,40) + eps)
Hypothesis: positive IC at 10d horizon.
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, "scripts")
from miner3_shared import load_close_panel, load_series, forward_return_panel, compute_ic, decay_profile
import scipy.stats as st

WINDOW = 40
END = "2035-12-18"
WATCH = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX",
         "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]

def metric(F, fwd):
    ics=[]; assets_ok=0
    for dt in fwd.index:
        if dt not in F.index: continue
        x = pd.concat([F.loc[dt], fwd.loc[dt]], axis=1).dropna()
        if len(x) < 8: continue
        if x.iloc[:,0].nunique() < 3 or x.iloc[:,1].nunique() < 3: continue
        ic,_ = st.spearmanr(x.iloc[:,0], x.iloc[:,1])
        if np.isnan(ic): continue
        ics.append(ic); assets_ok += len(x)
    ics = np.array(ics)
    if len(ics)==0: return None,0
    return {"ic":float(np.mean(ics)),"icir":float(np.mean(ics)/(np.std(ics)+1e-12)),
            "hit":float(np.mean(ics>0)),"n":len(ics),"avg":assets_ok/len(ics)}, len(ics)

# build factor
F = pd.DataFrame(index=None)
series={}
for a in WATCH:
    df0 = load_series(a)
    df0 = df0[df0.index <= pd.Timestamp(END)]
    hi = df0["high"].rolling(WINDOW).max()
    lo = df0["low"].rolling(WINDOW).min()
    rng = (hi - lo).replace(0, np.nan)
    series[a] = (df0["close"] - lo) / rng
F = pd.DataFrame(series)
F = F[F.index <= pd.Timestamp(END)]

close_panel = load_close_panel(END)
fwd10 = forward_return_panel(close_panel, 10)
r10, n = metric(F, fwd10)
print("FACTOR hi_lo_pos_40d  (close-min40)/(max40-min40)")
print("panel:", F.index.min().date(), "..", F.index.max().date(), "shape", F.shape)
print("coverage asset-days:", round(float(F.notna().sum().sum())/(F.shape[0]*F.shape[1]),3))
print("IC(10d):", round(r10["ic"],4) if r10 else None, "ICIR:", round(r10["icir"],4) if r10 else None,
      "hit:", round(r10["hit"],3) if r10 else None, "n_dates:", n, "avg assets:", round(r10["avg"],1) if r10 else None)

# decay
dec={}
for h in (1,2,3,5,10,20):
    fw = forward_return_panel(close_panel, h)
    rd,_ = metric(F, fw)
    dec[h] = round(rd["ic"],4) if rd else None
print("decay IC:", dec)

# subwindows
fwd = forward_return_panel(close_panel, 10)
for name, start in {"warm":"2020-01-01","2024+":"2024-01-01","2025+":"2025-01-01",
                    "2026+":"2026-01-01","online":"2026-07-16","2030+":"2030-01-01",
                    "2033+":"2033-01-01","recent":"2034-01-01"}.items():
    f = F[F.index >= pd.Timestamp(start)]; fw = fwd[fwd.index >= pd.Timestamp(start)]
    rm,_ = metric(f, fw)
    print(name, "IC:", round(rm["ic"],4) if rm else None, "ICIR:", round(rm["icir"],4) if rm else None)

# turnover (rank change fraction)
rr = F.rank(axis=1).diff().dropna()
turn = float((rr.abs()>2).mean(axis=1).mean()) if len(rr) else None
print("turnover(rank>2 chg fraction):", round(turn,4) if turn is not None else None)