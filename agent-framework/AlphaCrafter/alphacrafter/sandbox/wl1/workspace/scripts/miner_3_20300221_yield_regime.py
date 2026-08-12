import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
asof='2030-02-21'; syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=get_stock_daily_data(s,2800)
 if d is None or len(d)<150:d=get_index_daily_data(s,2800)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);d=d[d.date<=asof];px[s]=d.set_index('date').close.astype(float)
c=pd.DataFrame(px).sort_index(); r=c.pct_change(); spread=r['US10Y'].rolling(20).sum()-r['CN10Y'].rolling(20).sum()
# Cross-asset yield-spread regime: momentum is favored when the US/CN yield spread is stable or improving.
trend=c.pct_change(20); regime=np.tanh(spread/0.08)
f=trend*(1+0.35*regime.values[:,None]);f=pd.DataFrame(f,index=c.index,columns=c.columns).replace([np.inf,-np.inf],np.nan)
for h in [1,5,10,20]:
 fw=c.shift(-h)/c-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print(f'H={h} dates={len(q)} avg_n={q.n.mean():.2f} IC={q.ic.mean():.6f} ICIR={q.ic.mean()/q.ic.std(ddof=1):.6f} hit={(q.ic>0).mean():.4f}')
 for lab,a,b in [('2020-2025','2020','2025-12-31'),('2026-2028','2026','2028-12-31'),('2029','2029','2029-12-31'),('2030','2030','2030-02-21')]:
  x=q[(q.index>=a)&(q.index<=b)]
  if len(x):print(f' {lab}: dates={len(x)} IC={x.ic.mean():.6f} ICIR={x.ic.mean()/x.ic.std(ddof=1):.6f}')
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_3_20300221_yield_regime_signal.csv',index=False);print('coverage',out.symbol.nunique(),'rows',len(out),'turnover',f.rank(pct=True).diff().abs().mean().mean())
