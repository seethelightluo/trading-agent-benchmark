import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
 d=get_stock_daily_data(s,days=2800)
 if d is not None and len(d):
  x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); raw[s]=x.drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(raw).sort_index(); r=p.pct_change(); mom=p.pct_change(20); down=r.where(r<0).rolling(40,min_periods=15).std()
sig=(mom.sub(mom.median(axis=1),axis=0)/(down*np.sqrt(252)+.02)).shift(1)
for h in [1,5,10,20]:
 z=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],p.pct_change(h).shift(-h).loc[dt]],axis=1).dropna()
  if len(a)>=8:z.append((dt,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 z=pd.DataFrame(z,columns=['date','ic','n']).set_index('date')
 print(f'H={h} dates={len(z)} avg_n={z.n.mean():.2f} IC={z.ic.mean():.6f} ICIR={z.ic.mean()/z.ic.std(ddof=1):.6f} hit={(z.ic>0).mean():.3f}')
 for label,mask in [('2020-25',z.index<'2026-01-01'),('2026+',z.index>='2026-01-01'),('2028+',z.index>='2028-01-01'),('2029YTD',z.index>='2029-01-01')]:
  q=z.loc[mask]
  if len(q):print(f' {label}: n={len(q)} IC={q.ic.mean():.6f} ICIR={q.ic.mean()/q.ic.std(ddof=1):.6f}')
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20291018_defensive_relative_strength_signal.csv',index=False)
print('coverage=',sig.notna().mean().mean(),'dates=',len(sig),'instruments=',len(raw));print('turnover=',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
