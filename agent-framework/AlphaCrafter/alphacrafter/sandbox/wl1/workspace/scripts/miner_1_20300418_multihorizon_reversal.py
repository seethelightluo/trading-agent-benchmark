import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
asof='2030-04-17'; U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<150:d=get_index_daily_data(s,5000)
 if d is None:return None
 d=d.copy(); d.date=pd.to_datetime(d.date); return d[d.date<=asof].set_index('date').close.astype(float)
P=pd.DataFrame({s:load(s) for s in U}).sort_index(); r=P.pct_change(); vol=r.rolling(40,min_periods=25).std()*np.sqrt(40)
# Interpretable multi-horizon short-term reversal, volatility normalized and lagged.
f=-(0.50*r.rolling(3).sum()+0.30*r.rolling(5).sum()+0.20*r.rolling(10).sum())/(vol+0.01)
f=f.shift(1)
rowsall=[]
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); rowsall.append(q)
 print(f'H={h} dates={len(q)} avg_n={q.n.mean():.2f} IC={q.ic.mean():.6f} ICIR={q.ic.mean()/q.ic.std(ddof=1):.6f} hit={(q.ic>0).mean():.4f}')
 for a,b in [('2020','2025-12-31'),('2026','2028-12-31'),('2029','2029-12-31'),('2030','2030-04-17')]:
  x=q[(q.index>=a)&(q.index<=b)]
  if len(x):print(' ',a,len(x),f'IC={x.ic.mean():.6f}',f'ICIR={x.ic.mean()/x.ic.std(ddof=1):.6f}')
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_1_20300418_multihorizon_reversal_signal.csv',index=False)
print('assets',len(P.columns),'rows',len(out),'coverage',len(out)/((f.notna()).sum().sum()),'turnover',f.rank(pct=True).diff().abs().mean().mean())
