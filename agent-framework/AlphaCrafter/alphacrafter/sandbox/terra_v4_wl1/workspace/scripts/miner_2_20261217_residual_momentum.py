import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-16')
base='../persistent/stock_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'] for s in U}).sort_index().loc[:cut]
R=P.pct_change(); market=R.mean(axis=1)
# Residual momentum: asset cumulative 20d return minus contemporaneous cross-asset mean, risk scaled.
res=R.rolling(20).sum().sub(market.rolling(20).sum(),axis=0)
vol=R.rolling(20,min_periods=15).std()*np.sqrt(20)
F=res.div(vol.replace(0,np.nan))
for h in [1,5,10]:
 Y=P.shift(-h).div(P)-1; rows=[]
 for d in P.index:
  q=pd.concat([F.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y,method='spearman'),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print('H',h,'dates',len(ic),'avgN',round(a.n.mean(),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
 if h==1: print('years',[(int(y),round(g.mean(),6),len(g)) for y,g in ic.groupby(ic.index.year)])
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
F.to_csv('scripts/miner_2_20261217_residual_momentum_signal.csv',index_label='date')
print('assets',len(U),'rows',len(P),'cutoff',cut.date(),'artifact','scripts/miner_2_20261217_residual_momentum_signal.csv')
