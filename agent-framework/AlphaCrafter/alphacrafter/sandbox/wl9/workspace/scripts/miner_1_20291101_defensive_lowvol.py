import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];base='../persistent/stock_data';px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv');d.date=pd.to_datetime(d.date);px[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().loc[:'2029-10-31'];R=P.pct_change();
# Defensive score: inverse realized volatility, conditioned on positive medium trend.
v=R.rolling(30,min_periods=20).std()*np.sqrt(252); trend=P.pct_change(60); F=-(v) + 0.15*trend
print('candidate=defensive_lowvol_trend; universe',len(U),'dates',P.index.min().date(),P.index.max().date())
for h in [5,10,20,40]:
 fr=P.shift(-h).div(P)-1; vals=[];ds=[];ns=[]
 for dt in F.index:
  a=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: vals.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'));ds.append(dt);ns.append(len(a))
 x=np.array(vals);ok=np.isfinite(x);x=x[ok];ds=pd.DatetimeIndex(ds)[ok];sd=x.std(ddof=1)
 print('H',h,'dates',len(x),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',x.mean(),'ICIR',x.mean()/sd,'hit',np.mean(x>0),'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'valid',F.notna().mean().mean())
 for name,lo,hi in [('warmup','2020-01-01','2026-07-15'),('online','2026-07-16','2029-10-31'),('recent','2028-11-01','2029-10-31')]:
  z=x[(ds>=lo)&(ds<=hi)];print(name,len(z),z.mean() if len(z) else None,(z.mean()/z.std(ddof=1)) if len(z)>1 else None)
