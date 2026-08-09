import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut] for s in U}
p=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index(); o=pd.DataFrame({s:d.open for s,d in D.items()}).reindex(p.index)
r=p.pct_change(); gap=o/p.shift(1)-1
f=-gap.rolling(3,min_periods=3).mean(); f.to_csv('scripts/miner_1_20261217_gap_exhaustion_signal.csv')
rows=[]
for i in range(len(p)-1):
 q=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
  rows.append((p.index[i],q.f.corr(q.y,method='spearman'),len(q)))
a=pd.DataFrame(rows,columns=['date','ic','n']); x=a.ic
print('DATES',len(x),'AVG_N',round(a.n.mean(),2),'COVERAGE',round(a.n.mean()/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'HIT',round((x>0).mean(),4),'TURN',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 z=x[(pd.to_datetime(a.date).dt.year>=lo)&(pd.to_datetime(a.date).dt.year<=hi)]; print('REGIME',lo,hi,'N',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
for h in [1,5,10]:
 y=p.shift(-h).div(p)-1; vals=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1: vals.append(q.f.corr(q.y,method='spearman'))
 print('DECAY',h,'DATES',len(vals),'IC',round(np.mean(vals),6),'ICIR',round(np.mean(vals)/np.std(vals,ddof=1),6))
print('ARTIFACT','scripts/miner_1_20261217_gap_exhaustion_signal.csv')
