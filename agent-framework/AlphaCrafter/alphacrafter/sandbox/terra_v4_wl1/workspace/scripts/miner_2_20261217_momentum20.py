import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}; P=pd.DataFrame({s:D[s].close for s in U}).sort_index().loc[:cut]; P.index=pd.to_datetime(P.index)
# medium-term momentum, lagged at decision date
f=P.shift(1)/P.shift(21)-1
f.to_csv('scripts/miner_2_20261217_momentum20_signal.csv',index_label='date')
for h in [1,5,10]:
 y=P.shift(-h)/P-1; rows=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); a.index=pd.to_datetime(a.index); z=a.ic
 print('H',h,'dates',len(z),'avg_n',a.n.mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
 if h==1:
  for yr,g in z.groupby(z.index.year): print('YR',yr,len(g),g.mean(),g.mean()/g.std(ddof=1))
print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
