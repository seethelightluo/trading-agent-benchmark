import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:D[s].close for s in U}).sort_index().loc[:cut]
# Acceleration: recent 10d return relative to preceding 20d return, lagged one completed day.
r10=P.shift(1).div(P.shift(11))-1
r20prior=P.shift(1).div(P.shift(31))-1
f=(r10-r20prior).sub((r10-r20prior).median(axis=1),axis=0)
f.to_csv('scripts/miner_1_20261217_momentum_acceleration_signal.csv',index_label='date')
rows=[]
for h in [1,5,10]:
 y=P.shift(-h).div(P)-1; z=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8:z.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(z,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print('H',h,'dates',len(ic),'avg_n',a.n.mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
 if h==1:
  for name,g in ic.groupby(ic.index.year):print('YR',name,len(g),g.mean(),g.mean()/g.std(ddof=1))
print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
print('period',P.index.min(),P.index.max())
