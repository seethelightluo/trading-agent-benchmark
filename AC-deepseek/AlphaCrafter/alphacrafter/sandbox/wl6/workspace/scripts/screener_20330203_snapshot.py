import pandas as pd, numpy as np
VISIBLE="2033-02-02"
TRADABLE=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
OBS=['DXY','USDCNY','USDJPY','EURUSD','VIX']
def load(sym, ddir="../persistent/stock_data"):
    df=pd.read_csv(f"{ddir}/{sym}.csv", parse_dates=["date"])
    df=df[df["date"]<=pd.Timestamp(VISIBLE)].set_index("date").sort_index()
    return df
px=pd.DataFrame({s:load(s)["close"].astype(float) for s in TRADABLE})
obs={s:load(s,"../persistent/index_data")["close"].astype(float) for s in OBS}
ret=px.pct_change()
frozen=[s for s in TRADABLE if ret[s].dropna().iloc[-250:].abs().max()<1e-12 or px[s].nunique()<=1]
active=[s for s in TRADABLE if s not in frozen]
rows=[]
for s in TRADABLE:
    if s in frozen:
        rows.append([s,0,0,0,0,0,0,0.0]); continue
    r=px[s]
    rows.append([s, float(r.iloc[-1]/r.iloc[-6]-1), float(r.iloc[-1]/r.iloc[-22]-1), float(r.iloc[-1]/r.iloc[-62]-1),
                 float(r.iloc[-1]/r.iloc[-122]-1), float(ret[s].iloc[-20:].std()*np.sqrt(252)),
                 float(ret[s].iloc[-60:].std()*np.sqrt(252)), float(r.iloc[-1]/r.iloc[-60:].max()-1)])
tab=pd.DataFrame(rows,columns=["sym","r5","r20","r60","r120","vol20","vol60","dist60high"])
pd.set_option("display.width",200)
print("FROZEN:",frozen,"ACTIVE:",len(active))
print(tab.round(4).to_string(index=False))
print("\nOBSERVABLES:")
for o in OBS:
    v=obs[o]
    print(f"{o}: last={v.iloc[-1]:.2f} 20d={v.iloc[-1]/v.iloc[-21]-1:+.3f} 60d={v.iloc[-1]/v.iloc[-61]-1:+.3f} mean60={v.iloc[-60:].mean():.2f} min60={v.iloc[-60:].min():.2f} max60={v.iloc[-60:].max():.2f}")
spx=px['SPX']; spxr=ret['SPX']
ma20=spx.rolling(20).mean(); ma60=spx.rolling(60).mean(); ma120=spx.rolling(120).mean()
print("\nSPX trend: price=%.1f ma20=%.1f ma60=%.1f ma120=%.1f | ma20-slope=%+.3f ma60-slope=%+.3f | 60d=%+.3f 120d=%+.3f" % (
  spx.iloc[-1],ma20.iloc[-1],ma60.iloc[-1],ma120.iloc[-1],
  (ma20.iloc[-1]/ma20.iloc[-6]-1),(ma60.iloc[-1]/ma60.iloc[-6]-1),
  spx.iloc[-1]/spx.iloc[-61]-1, spx.iloc[-1]/spx.iloc[-121]-1))
print("SPX vol20=%.2f vol60=%.2f (ann. pct)" % (spxr.iloc[-20:].std()*np.sqrt(252)*100, spxr.iloc[-60:].std()*np.sqrt(252)*100))
above=[s for s in active if px[s].iloc[-1]>px[s].rolling(60).mean().iloc[-1]]
print("above 60d MA:",above,"|",len(above),"/",len(active))
# cross-sectional dispersion
xs=pd.DataFrame({s:px[s].iloc[-1]/px[s].iloc[-62]-1 for s in active}, index=[0]).iloc[0]
print("cross-sectional 60d return dispersion (std):", round(xs.std(),4))