import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
SYMS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fs={}
for s in SYMS:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<100: d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); fs[s]=d.set_index('date').sort_index()
p=pd.DataFrame({s:d.close for s,d in fs.items()})
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); v=v.set_index('date').close.sort_index().reindex(p.index).ffill()
r5=v.pct_change(5); z=(r5-r5.rolling(60,min_periods=30).mean())/(r5.rolling(60,min_periods=30).std()+1e-8)
amp=(1+z.clip(lower=0,upper=3)).rename('amp')
f=(-p.pct_change(5).mul(amp,axis=0)).shift(1)
print('symbols',len(fs),'date range',p.index.min(),p.index.max(),'coverage',p.notna().mean().mean())
for h in [5,10,20]:
 rows=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],p.pct_change(h).shift(-h).loc[dt]],axis=1).dropna()
  if len(q)>=8: rows.append((dt,len(q),q.iloc[:,0].corr(q.iloc[:,1],method='spearman')))
 r=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); ic=r.ic.mean(); ir=ic/r.ic.std(ddof=1)
 turn=f.rank(axis=1,pct=True).diff().abs().mean().mean()
 print('H',h,'dates',len(r),'avgN',r.n.mean(),'IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(ic,ir,(r.ic>0).mean(),turn))
 for lab,st in [('2026+','2026-01-01'),('2027+','2027-01-01'),('2028YTD','2028-01-01')]:
  x=r[r.date>=st].ic; print(lab,len(x),'IC %.6f ICIR %.6f'%(x.mean(),x.mean()/x.std(ddof=1)))
 r.to_csv('scripts/miner_1_20280601_vix_shock_reversal_%dd_signal.csv'%h,index=False)
print('max library corr unavailable: signal artifacts not loaded')
