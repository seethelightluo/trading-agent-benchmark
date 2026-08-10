import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
O=pd.DataFrame({s:D[s].open for s in U}).sort_index().loc[:cut]; C=pd.DataFrame({s:D[s].close for s in U}).reindex(O.index); H=pd.DataFrame({s:D[s].high for s in U}).reindex(O.index); L=pd.DataFrame({s:D[s].low for s in U}).reindex(O.index)
rng=(H-L).replace(0,np.nan)
# prior completed session: fade directional intraday displacement, normalized by range
f=(-(C-O)/rng).shift(1)
f.to_csv('scripts/miner_3_20261218_intraday_clv_reversal_signal.csv',index_label='date')
for h in [1,5,10]:
 y=C.shift(-h).div(C)-1; rows=[]
 for d in C.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print('H',h,'dates',len(ic),'avg_n',a.n.mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
 if h==1:
  for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
   z=ic.loc[lo:hi]; print('REG',lo,hi,len(z),z.mean(),z.mean()/z.std(ddof=1))
print('coverage',f.notna().sum().sum()/f.size,'period',C.index.min(),C.index.max())
