import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<100:d=get_index_daily_data(s,4000)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);D[s]=d.set_index('date')
px=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index();lr=np.log(px).diff()
disp=lr.std(axis=1).rolling(20,min_periods=10).mean(); med=disp.rolling(120,min_periods=40).median(); ds=(disp/med).clip(.5,2)
sig=(-lr.rolling(3).sum()/lr.rolling(20).std()*ds.shift(1)).shift(1);nxt=lr.shift(-1)
rows=[]
for dt in sig.index:
 a=sig.loc[dt]; b=nxt.loc[dt]; z=pd.concat([a.rename('a'),b.rename('b')],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.a.corr(z.b,method='spearman'),len(z)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print('assets',len(D),'dates',len(o),'avg_n',o.n.mean() if len(o) else 0,'coverage',o.n.sum()/(len(o)*len(U)) if len(o) else 0)
if len(o):
 print('IC %.8f ICIR %.8f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(ddof=1),(o.ic>0).mean()))
 for lab,sub in [('2020-22',o.loc['2020':'2022']),('2023-25',o.loc['2023':'2025']),('2026-27',o.loc['2026':'2027']),('2028-30',o.loc['2028':'2030']),('recent250',o.tail(250))]:
  if len(sub)>1: print(lab,len(sub),sub.ic.mean(),sub.ic.mean()/sub.ic.std(ddof=1),sub.n.mean())
o.reset_index().to_csv('scripts/miner_1_20300221_dispersion_reversal_signal.csv',index=False)
