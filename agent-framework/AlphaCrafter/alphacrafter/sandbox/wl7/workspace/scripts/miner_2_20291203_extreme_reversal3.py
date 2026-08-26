import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try:D[s]=get_index_daily_data(s,2500)
 except Exception:
  try:D[s]=get_stock_daily_data(s,2500)
  except Exception: D[s]=None
D={s:d for s,d in D.items() if d is not None and len(d)>100}
px=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items()})
ret=px.pct_change(); vol=ret.rolling(20).std()
base=-(px.shift(1)/px.shift(4)-1)/vol.shift(1)
loc=pd.DataFrame(index=px.index,columns=px.columns,dtype=float)
for s,d in D.items():
 x=d.set_index('date'); loc[s]=(x['close']-x['low'])/(x['high']-x['low']).replace(0,np.nan)
f=base*(0.5+abs(loc.shift(1)-0.5))
rows=[]
for i in range(len(px)-10):
 y=px.iloc[i+10]/px.iloc[i]-1; z=pd.concat([f.iloc[i],y],axis=1).dropna()
 if len(z)>=8: rows.append((px.index[i],z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('assets',len(D),'dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(),(r.ic>0).mean()))
for name,b in [('early','2024-01-01'),('middle','2027-01-01'),('late','2029-01-01')]:
 q=r.loc[b:]; print(name,len(q),q.ic.mean(),q.ic.mean()/q.ic.std())
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
r.to_csv('scripts/miner_2_20291203_extreme_reversal3_signal.csv')
print('artifact scripts/miner_2_20291203_extreme_reversal3_signal.csv')
