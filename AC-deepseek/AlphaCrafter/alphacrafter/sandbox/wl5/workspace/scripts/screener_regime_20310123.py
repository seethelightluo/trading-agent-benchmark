"""Screener regime assessment as of 2031-01-23 (visible through 2031-01-22)."""
import pandas as pd
import numpy as np

CUTOFF = "2031-01-22"
ASSETS = ["000300.SH","000688.SH","SPX","NDX","SOX","HSI","N225","SX5E",
          "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MACRO = ["VIX","DXY","USDCNY","USDJPY","EURUSD"]

def load(path, cutoff=CUTOFF):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= cutoff].reset_index(drop=True)
    return df

px = {}
for a in ASSETS:
    df = load(f"../persistent/stock_data/{a}.csv")
    px[a] = df.set_index("date")["close"]
pxdf = pd.DataFrame(px)

macro = {}
for m in MACRO:
    df = load(f"../persistent/index_data/{m}.csv")
    macro[m] = df.set_index("date")["close"]
macdf = pd.DataFrame(macro)

print("=== Last date in each series ===")
print(pxdf.index.max(), macdf.index.max())

# ---- Regime stats ----
print("\n=== 15-asset snapshot (pct moves) ===")
ret = pxdf.pct_change()
rows = {}
for a in ASSETS:
    c = pxdf[a]
    ma20 = c.rolling(20).mean().iloc[-1]
    ma60 = c.rolling(60).mean().iloc[-1]
    r5 = c.iloc[-1]/c.iloc[-6]-1
    r10 = c.iloc[-1]/c.iloc[-11]-1
    r20 = c.iloc[-1]/c.iloc[-21]-1
    r60 = c.iloc[-1]/c.iloc[-61]-1
    vol20 = ret[a].iloc[-20:].std()*np.sqrt(252)
    rows[a] = dict(last=round(c.iloc[-1],2), r5=round(r5*100,2), r10=round(r10*100,2),
                   r20=round(r20*100,2), r60=round(r60*100,2), vol20=round(vol20*100,1),
                   above_ma20=c.iloc[-1]>ma20, above_ma60=c.iloc[-1]>ma60)
snap = pd.DataFrame(rows).T
print(snap.to_string())

print("\n=== Macro snapshot ===")
mrows = {}
for m in MACRO:
    c = macdf[m]
    ma20 = c.rolling(20).mean().iloc[-1]
    ma60 = c.rolling(60).mean().iloc[-1]
    r10 = c.iloc[-1]/c.iloc[-11]-1
    r20 = c.iloc[-1]/c.iloc[-21]-1
    mrows[m] = dict(last=round(c.iloc[-1],2), r10=round(r10*100,2), r20=round(r20*100,2),
                    above_ma20=c.iloc[-1]>ma20, above_ma60=c.iloc[-1]>ma60)
print(pd.DataFrame(mrows).T.to_string())

# ---- Cross-sectional dispersion & avg correlation ----
print("\n=== Cross-sectional stats (last 60d) ===")
r60 = ret.iloc[-60:]
disp = r60.std(axis=1)  # cross-sectional dispersion of daily returns
print("avg cross-sectional daily disp (60d):", round(disp.mean()*100,3))
print("avg abs cross-sectional ret (60d):", round(r60.mean(axis=1).abs().mean()*100,3))
corr = ret.iloc[-60:].corr()
avg_corr = (corr.values.sum()-len(corr))/(len(corr)*(len(corr)-1))
print("avg pairwise corr (60d):", round(avg_corr,3))

# equity-block correlations
eq = ["SPX","NDX","SOX","HSI","N225","SX5E","000300.SH","000688.SH"]
print("avg equity-pair corr (60d):", round(((corr.loc[eq,eq].values.sum()-len(eq))/(len(eq)*(len(eq)-1))),3))

# ---- Trend strength proxies ----
print("\n=== Trend/MA structure (MA20 slope over 10d, in %) ===")
for a in ["WTI","SPX","NDX","SOX","BTC","ETH","XAU","COPPER","N225","SX5E","HSI"]:
    c = pxdf[a]
    ma20 = c.rolling(20).mean()
    slope = ma20.iloc[-1]/ma20.iloc[-11]-1
    r2 = None
    y = np.log(c.iloc[-30:].values)
    x = np.arange(len(y))
    if len(y) >= 18:
        beta, alpha = np.polyfit(x, y, 1)
        yhat = alpha + beta*x
        ss_res = np.sum((y-yhat)**2); ss_tot = np.sum((y-y.mean())**2)
        r2 = beta*np.sign(beta)*(ss_res/ss_tot if ss_tot>0 else 0)  # signed pseudo
    print(f"{a:8s} MA20slope10d={slope*100:+6.2f}%  last_above_MA20={c.iloc[-1]>ma20.iloc[-1]}")
