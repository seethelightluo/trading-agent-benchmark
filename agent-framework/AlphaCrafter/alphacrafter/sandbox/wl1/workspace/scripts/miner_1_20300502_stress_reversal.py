import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; asof='2030-05-01'
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<150:d=get_index_daily_data(s,5000)
 if d is None:return None
 d=d.copy();d.date=pd.to_datetime(d.date);return d[d.date<=asof].set_index('date').close.astype(float)
P=pd.DataFrame({s:load(s) for s in U}).sort_index();r=P.pct_change(); v=r.rolling(20,min_periods=15).std();
# volatility-conditioned short-term reversal: stronger reversal after unusually volatile shocks.
shock=r.rolling(3,min_periods=3).sum(); vr=v.div(v.rolling(60,min_periods=30).median()); f=(-shock/(v+0.01)*(1+0.5*vr.clip(0,3))).shift(1)
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(q,columns=['date','ic','n']).set_index('date');print(f'H={h} dates={len(q)} avgN={q.n.mean():.2f} IC={q.ic.mean():.6f} ICIR={q.ic.mean()/q.ic.std(ddof=1):.6f} hit={(q.ic>0).mean():.4f}')
 for a,b in [('2020','2025-12-31'),('2026','2028-12-31'),('2029','2029-12-31'),('2030','2030-05-01')]:
  x=q[(q.index>=a)&(q.index<=b)]
  if len(x):print(' ',a,len(x),f'IC={x.ic.mean():.6f}',f'ICIR={x.ic.mean()/x.ic.std(ddof=1):.6f}')
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_1_20300502_stress_reversal_signal.csv',index=False)
print('assets',len(P.columns),'coverage',len(out)/f.notna().sum().sum(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
