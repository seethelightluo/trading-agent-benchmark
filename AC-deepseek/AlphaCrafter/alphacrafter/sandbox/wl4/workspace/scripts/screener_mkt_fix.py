import pandas as pd, numpy as np

ASOF = "2034-03-20"
assets = ["000300.SH","000688.SH","SPX","NDX","SOX","HSI","N225","SX5E",
          "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]

def load(p):
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df

px = {}
for a in assets:
    df = load(f"../persistent/stock_data/{a}.csv")
    df = df[df.index <= ASOF]
    px[a] = df["close"].astype(float)
PX = pd.DataFrame(px)
RET = PX.pct_change()

liquid = [a for a in assets if PX[a].tail(60).nunique() > 2]
mkt_ew = (1+RET[liquid].mean(axis=1)).cumprod()   # equal-weight among liquid only
print("Equal-weight market (10 liquid assets, excluding frozen):")
for n,lab in [(5,"5d"),(10,"10d"),(21,"1m"),(63,"3m")]:
    if len(mkt_ew) > n:
        print(f"  {lab:>3}: {(mkt_ew.iloc[-1]/mkt_ew.iloc[-1-n]-1)*100:+.2f}%")
ma20 = mkt_ew.rolling(20).mean(); ma60 = mkt_ew.rolling(60).mean()
print(f"  price vs MA20: {(mkt_ew.iloc[-1]/ma20.iloc[-1]-1)*100:+.2f}% | vs MA60: {(mkt_ew.iloc[-1]/ma60.iloc[-1]-1)*100:+.2f}% | MA20>MA60: {ma20.iloc[-1]>ma60.iloc[-1]}")
rv20 = RET[liquid].mean(axis=1).tail(20).std()*np.sqrt(252)
rv60 = RET[liquid].mean(axis=1).tail(60).std()*np.sqrt(252)
print(f"  realized vol 20d: {rv20*100:.1f}% | 60d: {rv60*100:.1f}%")
dd = mkt_ew.iloc[-1]/mkt_ew.tail(252).max()-1
print(f"  drawdown from 1y high: {dd*100:+.2f}%")
# 10d and 21d returns of liquid names
r10 = (PX.iloc[-1]/PX.iloc[-11]-1)*100
r21 = (PX.iloc[-1]/PX.iloc[-22]-1)*100
r63 = (PX.iloc[-1]/PX.iloc[-64]-1)*100
tbl = pd.DataFrame({"r10":r10,"r21":r21,"r63":r63}).round(2)
print("\n", tbl.to_string())
print("\nEqual-weight mkt r21 (all 15 incl frozen at 0):", round(((1+RET.mean(axis=1)).tail(21).prod()-1)*100,2), "%")
