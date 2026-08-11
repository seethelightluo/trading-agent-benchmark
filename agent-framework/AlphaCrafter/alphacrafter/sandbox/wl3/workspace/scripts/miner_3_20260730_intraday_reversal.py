import pandas as pd, numpy as np
from scipy.stats import spearmanr
UNIV=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
frames={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date').loc[:cut] for s in UNIV}
def calc(h):
 rows=[]
 for s,d in frames.items():
  r=d.close.pct_change(); f=-(d.close/d.open-1)/r.rolling(20,min_periods=20).std(); y=d.close.shift(-h)/d.close-1
  rows.append(pd.DataFrame({'date':d.index,'f':f.values,'y':y.values}))
 q=pd.concat(rows,ignore_index=True).dropna(); vals=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8: vals.append(spearmanr(g.f,g.y).statistic)
 return np.array(vals),len(q),q.groupby('date').size().mean()
for h in [1,3,5,10]:
 a,n,av=calc(h); print('h',h,'dates',len(a),'observations',n,'avg_names',av,'IC %.8f ICIR %.8f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
a,n,av=calc(1)
for label,z in [('early',a[:len(a)//2]),('late',a[len(a)//2:]),('recent250',a[-250:])]: print(label,'ICIR %.8f'% (z.mean()/z.std(ddof=1)))
