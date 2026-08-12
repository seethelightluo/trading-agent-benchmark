import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=3200)
 if d is None or len(d)<200: d=get_index_daily_data(s,days=3200)
 if d is not None and len(d)>200:
  x=d.copy(); x.date=pd.to_datetime(x.date); P[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill(); r=np.log(p).diff()
# Momentum curvature: recent 5-session return relative to a quarter of 20-session return,
# scaled by trailing 30-session realized volatility; all inputs are completed-day data.
vol=r.rolling(30).std()*np.sqrt(5)
f=(r.rolling(5).sum()-0.25*r.rolling(20).sum())/(vol+1e-12)
f=f.replace([np.inf,-np.inf],np.nan); f=f.sub(f.mean(axis=1),axis=0)
def calc(h):
 y=np.log(p.shift(-h)/p); out=[]
 for dt in f.index:
  x=f.loc[dt]; z=y.loc[dt]; ok=x.notna()&z.notna()
  if ok.sum()>=8: out.append((dt,x[ok].corr(z[ok],method='spearman'),ok.sum()))
 a=pd.DataFrame(out,columns=['date','ic','n']).dropna(); return a
for h in [1,3,5,10]:
 a=calc(h); print('H',h,'dates',len(a),'avgN',round(a.n.mean(),2),'IC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/a.ic.std(ddof=1),6),'ICIR_sqrt',round(a.ic.mean()/a.ic.std(ddof=1)*np.sqrt(len(a)),6),'hit',round((a.ic>0).mean(),4))
 if h==5:
  for lo,hi in [(2020,2022),(2023,2025),(2026,2027),(2028,2029)]:
   q=a[(a.date.dt.year>=lo)&(a.date.dt.year<=hi)]; print('REG',lo,hi,len(q),round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(ddof=1),6) if len(q)>1 else np.nan)
print('coverage',f.notna().sum().sum()/(f.shape[0]*len(U)),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean(),'assets',len(P),'dates',len(f))
f.to_csv('scripts/miner_1_20290920_momentum_curvature_signal.csv',index_label='date')
