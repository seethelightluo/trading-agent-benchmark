import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4500)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date)
  C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
px=pd.DataFrame(C).sort_index(); r=px.pct_change()
# Cross-asset breadth and volatility-compression trend. All inputs are known at date t.
mom=px.pct_change(20)
vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
breadth=(mom>0).mean(axis=1)
# high breadth rewards continuation; low breadth reverses the trend modestly
f=(mom/vol.replace(0,np.nan)).mul(0.5+0.5*breadth,axis=0)
f=f.replace([np.inf,-np.inf],np.nan)
fwd=px.shift(-10)/px-1
rows=[]; sig=[]
for dt in f.index:
 a=f.loc[dt]; y=fwd.loc[dt]; ok=a.notna()&y.notna()
 if ok.sum()>=8:
  rows.append((dt,a[ok].corr(y[ok],method='spearman'),int(ok.sum())))
  for s in U:
   if s in f.columns: sig.append({'date':dt,'symbol':s,'signal':a.get(s,np.nan)})
x=pd.DataFrame(rows,columns=['date','ic','n']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('dates',len(x),'mean_n',round(x.n.mean(),2),'coverage',round(x.n.sum()/(len(x)*len(U)),6))
print('IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((x.ic>0).mean(),6))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2032-09-15')]:
 z=x[(x.date>=a)&(x.date<=b)]; print('regime',a,b,'dates',len(z),'IC',round(z.ic.mean(),6) if len(z) else None,'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),6) if len(z)>1 else None)
q=f.rank(axis=1,pct=True); turnover=q.diff().abs().mean(axis=1).mean(); print('turnover',round(float(turnover),6))
pd.DataFrame(sig).to_csv('scripts/miner_2_20320916_breadth_scaled_trend_signal.csv',index=False)
print('artifact','scripts/miner_2_20320916_breadth_scaled_trend_signal.csv')
