import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv');d.date=pd.to_datetime(d.date);px[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().loc[:'2029-11-14'];R=P.pct_change()
# Stable medium trend: 60d return divided by total absolute daily movement, with a mild inverse-volatility overlay.
ret60=P.pct_change(60); efficiency=ret60/(R.abs().rolling(60,min_periods=45).sum())
vol30=R.rolling(30,min_periods=20).std()*np.sqrt(252)
F=efficiency/(vol30+1e-9)
print('candidate=path_quality_defensive; instruments',len(U),'dates',len(P),'range',P.index.min().date(),P.index.max().date())
for h in [5,10,20,40]:
 fr=P.shift(-h).div(P)-1; vals=[];ds=[];ns=[]
 for dt in F.index:
  a=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: vals.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'));ds.append(dt);ns.append(len(a))
 x=np.asarray(vals);ok=np.isfinite(x);x=x[ok];ds=pd.DatetimeIndex(ds)[ok];
 print('H',h,'dates',len(x),'avg_n',round(float(np.mean(np.asarray(ns)[ok])),2),'coverage',round(float(np.mean(np.asarray(ns)[ok])/15),4),'IC',round(float(x.mean()),6),'ICIR',round(float(x.mean()/x.std(ddof=1)),6),'hit',round(float(np.mean(x>0)),4))
 for name,lo,hi in [('warmup','2020-01-01','2026-07-15'),('online','2026-07-16','2029-11-14'),('recent','2028-11-15','2029-11-14')]:
  z=x[(ds>=lo)&(ds<=hi)];print(name,'n',len(z),'IC',round(float(z.mean()),6) if len(z) else None,'ICIR',round(float(z.mean()/z.std(ddof=1)),6) if len(z)>1 else None)
rank=F.rank(axis=1,pct=True);print('turnover',float(rank.diff().abs().mean(axis=1).mean()),'valid_coverage',float(F.notna().mean().mean()))
