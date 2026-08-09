import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:D[s]['close'] for s in U}).sort_index().loc[:cut]
r=P.pct_change()
# Range-efficiency trend: signed net move divided by path length, lagged one day.
# This rewards persistent directional travel and discounts choppy returns.
f=(P.pct_change(20)/(r.abs().rolling(20,min_periods=15).sum())).shift(1)
f.to_csv('scripts/miner_1_20261217_range_efficiency_signal.csv',index_label='date')
for h in [1,5,10]:
 y=P.shift(-h).div(P)-1; rows=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y,method='spearman'),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); x=a.ic
 print('H',h,'dates',len(x),'avgN',round(a.n.mean(),2),'IC',round(x.mean(),8),'ICIR',round(x.mean()/x.std(ddof=1),8),'hit',round((x>0).mean(),4))
 for yr,g in x.groupby(x.index.year): print('YR',yr,'n',len(g),'IC',round(g.mean(),6),'ICIR',round(g.mean()/g.std(ddof=1),5))
print('coverage',round(f.notna().sum().sum()/f.size,6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6),'period',P.index.min(),P.index.max())
