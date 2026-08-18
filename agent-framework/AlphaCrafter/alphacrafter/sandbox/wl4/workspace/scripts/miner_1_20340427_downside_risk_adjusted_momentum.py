import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_index_daily_data(s,4000)
    except Exception: x=None
    if x is None or len(x)<300:
        try: x=get_stock_daily_data(s,4000)
        except Exception: x=None
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); D[s]=x.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
mom=p.shift(1).pct_change(30); down=r.where(r<0,0).rolling(30,min_periods=20).std().shift(1)
f=(mom/(down+1e-8)).replace([np.inf,-np.inf],np.nan)
fr=p.shift(-10)/p-1; rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt].rename('factor'),fr.loc[dt].rename('forward')],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.factor.corr(z.forward),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('assets',len(D),'dates',len(p),'IC_dates',len(q),'avg_n',q.n.mean())
print('IC %.6f ICIR %.6f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1), (q.ic>0).mean()))
for n in [260,520,1040,1560]:
 z=q.tail(n); print('recent',n,'dates',len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1), (z.ic>0).mean()))
cov=f.notna().sum(axis=1).mean()/len(U); rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).mean()
print('coverage %.4f turnover %.4f'% (cov,turn))
for h in [1,5,10,20]:
 ff=p.shift(-h)/p-1; rr=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('factor'),ff.loc[dt].rename('forward')],axis=1).dropna()
  if len(z)>=8: rr.append(z.factor.corr(z.forward))
 print('decay',h,'IC %.6f'%np.nanmean(rr),'n',len(rr))
