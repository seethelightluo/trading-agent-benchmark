import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:D[s].close for s in U}).sort_index().loc[:cut]
# Peer-relative 5-day reversal, entirely lagged: asset return less same-day cross-sectional peer median.
r5=P.shift(1).div(P.shift(6))-1
peer=r5.apply(lambda x:(x.sum()-x)/(x.notna().sum()-1),axis=1)
f=-(r5-peer)
f.to_csv('scripts/miner_1_20261217_peer_relative_reversal_signal.csv',index_label='date')
for h in [1,5,10]:
 y=P.shift(-h).div(P)-1; rows=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print('H',h,'dates',len(ic),'avg_n',round(a.n.mean(),2),'IC',round(ic.mean(),8),'ICIR',round(ic.mean()/ic.std(ddof=1),8),'hit',round((ic>0).mean(),4))
 if h==1:
  for yr,g in ic.groupby(ic.index.year): print('YR',yr,'dates',len(g),'IC',round(g.mean(),8),'ICIR',round(g.mean()/g.std(ddof=1),8))
print('coverage',round(f.notna().sum().sum()/f.size,6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
print('period',P.index.min().date(),P.index.max().date(),'assets',len(U))
