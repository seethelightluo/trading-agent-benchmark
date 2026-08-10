import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); b='../persistent/stock_data'
D={s:pd.read_csv(f'{b}/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
O=pd.DataFrame({s:D[s].open for s in U}).sort_index().loc[:cut]; C=pd.DataFrame({s:D[s].close for s in U}).reindex(O.index)
# fade overnight gap, using close/open from prior completed day
f=-(O/C.shift(1)-1).shift(1)
f.to_csv('scripts/miner_3_20261218_overnight_gap_signal.csv',index_label='date')
for h in [1,5,10]:
 y=C.shift(-h).div(C)-1; z=[]
 for d in C.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8:z.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(z,columns=['date','ic','n']).set_index('date'); x=a.ic
 print(h,len(x),a.n.mean(),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean())
 if h==1:
  for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
   q=x.loc[lo:hi]; print(lo,len(q),q.mean(),q.mean()/q.std(ddof=1))
print('coverage',f.notna().sum().sum()/f.size)
