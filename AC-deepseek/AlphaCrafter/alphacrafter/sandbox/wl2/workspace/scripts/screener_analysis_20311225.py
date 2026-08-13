"""Screener analysis for cycle at 2031-12-25 (visible through 2031-12-24)."""
import json, math
import numpy as np
import pandas as pd

VISIBLE = "2031-12-24"
ASSETS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU",
          "COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
FLAT_FEED = {"HSI","SX5E","BTC","US10Y","CN10Y"}

def load(name, path=None):
    p = path or f"../persistent/stock_data/{name}.csv"
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= VISIBLE].set_index("date")
    return df

closes = {}
for a in ASSETS:
    df = load(a)
    closes[a] = df["close"].astype(float)
    print(a, len(df), df.index[-1].date(), round(float(df["close"].iloc[-1]), 4))

px = pd.DataFrame(closes).dropna(how="all")
rets = px.pct_change()

# ---- returns across horizons ----
print("\n=== RETURNS (as of 2031-12-24) ===")
rows = {}
for a in ASSETS:
    c = px[a].dropna()
    if len(c) < 190:
        continue
    r = {}
    for n in (5, 10, 20, 60, 120, 180):
        r[f"r{n}"] = float(c.iloc[-1] / c.iloc[-1-n] - 1.0) if len(c) > n else np.nan
    # 252d range position
    lo = c.tail(252).min(); hi = c.tail(252).max()
    r["range252"] = float((c.iloc[-1]-lo)/(hi-lo)) if hi>lo else np.nan
    # vol
    s = rets[a].dropna().tail(20)
    r["vol20_ann"] = float(s.std()*math.sqrt(252)) if len(s)>=10 else np.nan
    # 60d downbeta to SPX
    sub = pd.concat([rets[a], rets["SPX"]], axis=1, join="inner").dropna().tail(60)
    spxr = sub.iloc[:,1]
    dn = sub[spxr < 0]
    r["downbeta60"] = float(dn.iloc[:,0].cov(dn.iloc[:,1])/dn.iloc[:,1].var()) if (len(dn)>=15 and dn.iloc[:,1].var()>1e-12) else np.nan
    # 60d corr to SPX
    r["corr60"] = float(sub.iloc[:,0].corr(sub.iloc[:,1])) if len(sub)>=15 else np.nan
    # max consecutive gain streak (21d window)
    pos = (rets[a] > 0).astype(int).tail(21)
    m=cur=0
    for v in pos:
        if v==1: cur+=1; m=max(m,cur)
        else: cur=0
    r["max_consec_gain"] = m
    # 180d skip5 momentum
    r["mom180"] = float(c.iloc[-1]/c.iloc[-186]-1.0) if len(c)>186 else np.nan
    # flat feed guard
    r["flat15"] = bool(len(s)>=15 and float(rets[a].dropna().tail(15).std())<1e-12)
    rows[a]=r

tab = pd.DataFrame(rows).T
pd.set_option("display.width", 250)
print(tab.round(4).to_string())

# ---- VIX / macro ----
print("\n=== MACRO ===")
for name in ["VIX","DXY","USDJPY","EURUSD","USDCNY"]:
    try:
        df = pd.read_csv(f"../persistent/index_data/{name}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= VISIBLE].set_index("date")
        c = df["close"].astype(float)
        r5 = float(c.iloc[-1]/c.iloc[-6]-1.0) if len(c)>6 else np.nan
        r20 = float(c.iloc[-1]/c.iloc[-21]-1.0) if len(c)>21 else np.nan
        r60 = float(c.iloc[-1]/c.iloc[-61]-1.0) if len(c)>61 else np.nan
        print(name, "last", round(float(c.iloc[-1]),2), "5d", round(r5*100,2), "20d", round(r20*100,2), "60d", round(r60*100,2))
    except Exception as e:
        print(name, "ERR", e)

# ---- dispersion ----
print("\n=== DISPERSION (cross-sectional) ===")
r20 = tab["r20"].dropna()
r60 = tab["r60"].dropna()
print("20d spread pp:", round((r20.max()-r20.min())*100,1), "| 60d spread pp:", round((r60.max()-r60.min())*100,1))
print("20d cross-sectional std:", round(r20.std()*100,2))
# pairwise corr of 10 live names
live = [a for a in ASSETS if a not in FLAT_FEED]
rr = rets[live].tail(60)
cm = rr.corr()
vals = cm.values[np.triu_indices_from(cm.values, k=1)]
print("mean pairwise corr (60d, 10 live):", round(float(np.nanmean(vals)),4))

# ---- factor q ranking ----
print("\n=== FACTOR QUALITY (persisted validation metrics) ===")
qdata = {
 "max_consec_gain_20": (0.0682, 0.231, 0.2318, 0.7238),
 "mom_180d_skip5":     (0.0495, 0.125, 0.2680, 0.4420),
 "downbeta_spx_60":    (0.0752, 0.1871, 0.0720, 0.6473),
 "spx_corr60":         (0.0558, 0.1556, 0.0612, 0.9869),
 "range_pos_252":      (0.0355, 0.1070, 0.1533, 0.7158),
}
for k,(ic,icir,tov,cov) in qdata.items():
    print(k, "q=", round(abs(ic)*abs(icir),6), "IC", ic, "ICIR", icir, "turnover", tov, "coverage", cov)

# regime risk score (v6 formula)
m20 = float(np.nanmean([tab.loc[a,"r20"] for a in ASSETS if a not in FLAT_FEED and not math.isnan(tab.loc[a,"r20"])]))
print("\nmean live 20d ret:", round(m20*100,2), "%")
