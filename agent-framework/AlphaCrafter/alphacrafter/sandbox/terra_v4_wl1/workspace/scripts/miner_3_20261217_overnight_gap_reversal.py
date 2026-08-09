import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:D[s].close for s in U}).sort_index().loc[:cut]
O=pd.DataFrame({s:D[s].open for s in U}).reindex(P.index)
# Fade completed overnight gap, with robust clipping to avoid single-name shock dominance.
gap=(O/P.shift(1)-1).clip(-.15,.15); F=-gap
for h in [1,5,10]:
 Y=P.shift(-h)/P-1; rows=[]
 for d in P.index:
  q=pd.concat([F.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y),len(q)))
 A=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=A.ic
 print('H',h,'dates',len(ic),'avg_n',round(A.n.mean(),2),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
 if h==1:
  for yr,g in ic.groupby(ic.index.year): print('YR',yr,len(g),g.mean(),g.mean()/g.std(ddof=1))
print('coverage',F.notna().sum().sum()/F.size,'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
print('period',P.index.min(),P.index.max())
# save artifact for deterministic audit
out=pd.DataFrame(F.stack(),columns=['signal']); out.index.names=['date','symbol']; out.to_csv('scripts/miner_3_20261217_overnight_gap_reversal_signal.csv')
