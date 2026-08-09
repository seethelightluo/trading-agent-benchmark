import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cut=pd.Timestamp('2026-12-17')
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}; P=pd.DataFrame({s:D[s].close for s in U}).sort_index().loc[:cut]
VIX=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().reindex(P.index).close
r20=P.shift(1).div(P.shift(21))-1; vol=P.pct_change().rolling(20,min_periods=15).std().shift(1)
# In risk-off conditions, prefer recent losers; in calm conditions, favor trend. VIX is lagged and cross-section demeaned.
z=(VIX.shift(1)>VIX.shift(1).rolling(60,min_periods=30).median()).astype(float)
f=(-(r20/vol)*(2*z-1)).replace([np.inf,-np.inf],np.nan)
f.to_csv('scripts/miner_1_20261217_vix_conditioned_momentum_signal.csv',index_label='date')
for h in [1,5,10]:
 y=P.shift(-h).div(P)-1; rows=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); a.index=pd.to_datetime(a.index); ic=a.ic
 print('H',h,'dates',len(ic),'avg_n',round(a.n.mean(),2),'IC',round(ic.mean(),8),'ICIR',round(ic.mean()/ic.std(ddof=1),8),'hit',round((ic>0).mean(),4))
 if h==1:
  for yr,g in ic.groupby(ic.index.year): print('YR',yr,len(g),round(g.mean(),8),round(g.mean()/g.std(ddof=1),8))
print('coverage',round(f.notna().sum().sum()/f.size,6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
