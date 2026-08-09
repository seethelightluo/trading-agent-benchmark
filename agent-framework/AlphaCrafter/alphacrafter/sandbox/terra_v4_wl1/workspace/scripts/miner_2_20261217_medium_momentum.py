import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:D[s].close for s in U}).sort_index().loc[:cut]
r=P.pct_change(); vol=r.rolling(20,min_periods=15).std()
# Short history relative to the current simulated date; use all valid rows and a 60-session lookback.
f=(P.shift(1)/P.shift(61)-1).div(vol.shift(1)).replace([np.inf,-np.inf],np.nan)
f.to_csv('scripts/miner_2_20261217_medium_momentum_signal.csv',index_label='date')
for h in [1,5,10]:
 y=P.shift(-h).div(P)-1; rows=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=a.ic
 print('H',h,'dates',len(z),'avg_n',round(a.n.mean(),2),'IC',round(z.mean(),5),'ICIR',round(z.mean()/z.std(ddof=1),5),'hit',round((z>0).mean(),4))
 if h==1:
  for yr,g in z.groupby(z.index.year): print('YR',yr,len(g),round(g.mean(),5),round(g.mean()/g.std(ddof=1),4))
print('coverage',round(f.notna().sum().sum()/f.size,5),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
print('valid dates by n>=8',sum(f.notna().sum(axis=1)>=8),'period',P.index.min().date(),P.index.max().date())
