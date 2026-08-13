import pandas as pd, numpy as np, os

assets = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
base = "../persistent/stock_data"
rows = []
for a in assets:
    df = pd.read_csv(os.path.join(base, a+".csv"))
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    df = df[df["date"] <= "2031-05-08"]
    if len(df) < 70:
        rows.append(dict(asset=a, note="insufficient data", n=len(df)))
        continue
    px = df["close"].astype(float)
    last = px.iloc[-1]
    def r(d):
        return (px.iloc[-1]/px.iloc[-1-d]-1)*100 if len(px)>d else np.nan
    ma20 = px.rolling(20).mean().iloc[-1]
    ma60 = px.rolling(60).mean().iloc[-1]
    vol20 = px.pct_change().rolling(20).std().iloc[-1]*np.sqrt(252)*100
    dd = (px.iloc[-60:]/px.iloc[-60:].cummax()-1).min()*100
    ret5,ret10,ret20,ret40,ret60 = r(5),r(10),r(20),r(40),r(60)
    rows.append(dict(asset=a, close=round(last,2), r5=round(ret5,2) if ret5==ret5 else None,
                     r10=round(ret10,2) if ret10==ret10 else None, r20=round(ret20,2) if ret20==ret20 else None,
                     r40=round(ret40,2) if ret40==ret40 else None, r60=round(ret60,2) if ret60==ret60 else None,
                     above_ma20 = bool(last>ma20), above_ma60=bool(last>ma60), ma20_gap=round((last/ma20-1)*100,2),
                     vol20=round(vol20,1), dd60=round(dd,1)))
out = pd.DataFrame(rows)
print(out.to_string(index=False))
print()
print("Breadth above MA20:", int(out.above_ma20.sum()), "/", len(out))
print("Breadth above MA60:", int(out.above_ma60.sum()), "/", len(out))
print("Avg 20d ret:", round(out.r20.mean(),2), "median:", round(out.r20.median(),2))
print("Avg 10d ret:", round(out.r10.mean(),2), "median:", round(out.r10.median(),2))
print("Avg 60d ret:", round(out.r60.mean(),2), "median:", round(out.r60.median(),2))
print("Mean vol20:", round(out.vol20.mean(),1))
