import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut] for s in U}
O=pd.DataFrame({s:D[s].open for s in U}); C=pd.DataFrame({s:D[s].close for s in U}); prev=C.shift(1)
gap=O/prev-1
# Fade lagged overnight gap, with robust cross-sectional clipping
f=-gap.shift(1)
f=f.clip(f.quantile(.05,axis=1),f.quantile(.95,axis=1),axis=0)
for h in [1,5,10]:
 y=C.shift(-h)/C-1; rows=[]
 for d in C.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=a.ic
 print('H',h,'dates',len(z),'avgN',a.n.mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
 if h==1:
  for yr,g in z.groupby(z.index.year): print('YR',yr,len(g),g.mean(),g.mean()/g.std(ddof=1))
print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
f.to_csv('scripts/miner_2_20261217_overnight_gap_reversal_signal.csv',index_label='date'); print('ARTIFACT scripts/miner_2_20261217_overnight_gap_reversal_signal.csv')
