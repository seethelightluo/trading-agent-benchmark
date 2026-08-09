import numpy as np, pandas as pd
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
D={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
 D[s]=d
idx=pd.Index(sorted(set().union(*[set(x.index) for x in D.values()])))
C=pd.DataFrame({s:D[s]['close'].reindex(idx) for s in U}); H=pd.DataFrame({s:D[s]['high'].reindex(idx) for s in U}); L=pd.DataFrame({s:D[s]['low'].reindex(idx) for s in U}); O=pd.DataFrame({s:D[s]['open'].reindex(idx) for s in U})
# 3-day smoothed close-location value, range-normalized intraday pressure
rng=(H-L).replace(0,np.nan)
clv=((C-L)/rng*2-1)
f=clv.rolling(3,min_periods=2).mean()
# use only completed date; forward close return
for h in [1,5,10]:
 y=C.shift(-h).div(C)-1; rows=[]
 for d in idx:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=a.ic
 print('H',h,'dates',len(z),'avg_n',round(a.n.mean(),2),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
 if h==1:
  for yr,g in z.groupby(z.index.year): print('YR',yr,len(g),round(g.mean(),5),round(g.mean()/g.std(ddof=1),4))
print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
f.to_csv('scripts/miner_1_20261217_clv3_signal.csv',index_label='date')
print('ARTIFACT scripts/miner_1_20261217_clv3_signal.csv')
