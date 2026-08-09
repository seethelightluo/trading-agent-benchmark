import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-16'); base='../persistent/stock_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'] for s in U}).sort_index().loc[:cut]
# Trend acceleration: recent 5d return relative to its average daily 20d trend.
r5=P.pct_change(5); r20=P.pct_change(20)
F=r5-r20/4
P.to_csv('/tmp/unused.csv')
for h in [1,5,10]:
 Y=P.shift(-h).div(P)-1; rows=[]
 for d in P.index:
  q=pd.concat([F.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((d,q.f.corr(q.y,method='spearman'),len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print('H',h,'dates',len(ic),'avgN',round(a.n.mean(),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
 if h==1: print('years',[(int(y),round(g.mean(),6),len(g)) for y,g in ic.groupby(ic.index.year)])
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
# correlation with existing plain 5d reversal
z=pd.concat([F.stack().rename('f'),(-r5).stack().rename('rev')],axis=1).dropna(); print('corr_plain_reversal',round(z.f.corr(z.rev),6))
F.to_csv('scripts/miner_2_20261217_trend_acceleration_signal.csv',index_label='date')
print('assets',len(U),'rows',len(P),'cutoff',cut.date(),'artifact','scripts/miner_2_20261217_trend_acceleration_signal.csv')
