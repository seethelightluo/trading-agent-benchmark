import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-07-15')
# causal close-to-open gap reversal, evaluated against next completed daily return
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); d=d[d.date<=END].set_index('date')
 gap=d.open/d.close.shift(1)-1; f=-gap; y=d.close.pct_change().shift(-1)
 q=pd.concat([f,y],axis=1); q.columns=['f','y']; q['date']=q.index; rows.append(q.reset_index(drop=True))
a=pd.concat(rows,ignore_index=True).dropna(); obs=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:obs.append((dt,spearmanr(g.f,g.y).statistic,len(g)))
o=pd.DataFrame(obs,columns=['date','ic','n']).dropna(); print('candidate=gap_reversal_1d dates',len(o),'avg_n',o.n.mean(),'coverage',len(o)/a.date.nunique(),'IC %.8f ICIR %.8f hit %.5f'%(o.ic.mean(),o.ic.mean()/o.ic.std(ddof=1),(o.ic>0).mean()))
for h in [5,10]:
 rows=[]
 for s in U:
  d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); d=d[d.date<=END].set_index('date'); f=-(d.open/d.close.shift(1)-1); y=d.close.pct_change(h).shift(-h)
  q=pd.concat([f,y],axis=1); q.columns=['f','y']; q['date']=q.index; rows.append(q.reset_index(drop=True))
 b=pd.concat(rows,ignore_index=True).dropna(); z=[]
 for dt,g in b.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:z.append(spearmanr(g.f,g.y).statistic)
 z=pd.Series(z).dropna(); print('decay',h,'IC %.8f ICIR %.8f dates %d'%(z.mean(),z.mean()/z.std(ddof=1),len(z)))
r=a.copy(); r['rank']=r.groupby('date').f.rank(pct=True); r=r.sort_values(['date','date']); print('assets',len(U),'period',a.date.min(),a.date.max())
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 x=o[(o.date.dt.year>=lo)&(o.date.dt.year<=hi)].ic; print('regime',lo,hi,'IC %.8f ICIR %.8f dates %d'%(x.mean(),x.mean()/x.std(ddof=1),len(x)))
