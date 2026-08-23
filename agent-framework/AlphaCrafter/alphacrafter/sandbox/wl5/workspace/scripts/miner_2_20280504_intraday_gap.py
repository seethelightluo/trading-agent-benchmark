import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; H=10
F={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); F[s]=d.drop_duplicates('date').set_index('date').sort_index()
o=pd.DataFrame({s:d.open for s,d in F.items()}); c=pd.DataFrame({s:d.close for s,d in F.items()});
# intraday return, reverse unusually large moves relative to cross-sectional median
x=o/c.shift(1)-1
med=x.median(axis=1); mad=x.sub(med,axis=0).abs().median(axis=1).replace(0,np.nan)
sig=-x.sub(med,axis=0).div(mad,axis=0).clip(-5,5)
fwd=c.shift(-H)/c-1
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('idea=intraday_gap_relative_reversal horizon=10'); print('dates',len(r),'avg_n',round(r.n.mean(),3),'coverage',round(r.n.sum()/(len(r)*15),4)); print('IC %.6f ICIR %.6f hit %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1),(r.ic>0).mean()))
for a,b in [('2020','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31')]:
 q=r.loc[a:b]
 if len(q): print(a+'..'+b,'dates',len(q),'IC %.6f ICIR %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
rank=sig.rank(axis=1,pct=True); turn=rank.diff().abs().stack().groupby(level=0).mean().reindex(r.index).mean(); print('turnover_proxy',round(float(turn),6))
sig.loc[r.index].rename_axis('date').reset_index().to_csv('scripts/miner_2_20280504_intraday_gap_signal.csv',index=False); print('artifact scripts/miner_2_20280504_intraday_gap_signal.csv')
