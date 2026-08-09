import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2027-02-25'); rows=[]
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date <= @end').sort_values('date')
 d['gap']=d.open/d.close.shift(1)-1; d['asset']=a
 for h in [1,5,10]: d[f'fwd{h}']=d.close.shift(-h)/d.close-1
 rows.append(d[['date','asset','gap','fwd1','fwd5','fwd10']])
x=pd.concat(rows)
for h in [1,5,10]:
 vals=[]
 for dt,g in x.groupby('date'):
  g=g.replace([np.inf,-np.inf],np.nan).dropna(subset=['gap',f'fwd{h}'])
  if len(g)>=8 and g.gap.nunique()>1 and g[f'fwd{h}'].nunique()>1:
   vals.append((dt,spearmanr(-g.gap,g[f'fwd{h}']).statistic,len(g)))
 z=np.array([v for _,v,_ in vals]); print('h',h,'dates',len(z),'avg_n',np.mean([n for _,_,n in vals]),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0))
 for y in range(2020,2028):
  q=np.array([v for dt,v,n in vals if dt.year==y]);
  if len(q):print(y,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),4))
 if h==1:
  out=x.pivot(index='date',columns='asset',values='gap').loc[:end]; (-out).to_csv('../persistent/factor_signals_miner_1_20270225_overnight_gap_reversal.csv')
