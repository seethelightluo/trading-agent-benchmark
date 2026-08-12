import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date <= @cut').set_index('date').sort_index() for s in U}
for k in [2,3,5,10]:
 rows=[]
 for s,x in D.items():
  r=x.close.pct_change();v=r.rolling(20,min_periods=10).std();raw=-r.rolling(3,min_periods=3).sum()/v
  f=raw.where(np.arange(len(x))%k==0).ffill()
  rows.append(pd.DataFrame({'date':x.index,'f':f,'y':r.shift(-1),'symbol':s}).reset_index(drop=True))
 a=pd.concat(rows,ignore_index=True).dropna();out=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1:
   c=spearmanr(g.f,g.y).statistic
   if pd.notna(c):out.append(c)
 o=pd.Series(out); ranks=a.assign(rank=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='symbol',values='rank')
 print('hold',k,'dates',len(o),'avg_n',a.groupby('date').size().mean(),'IC',o.mean(),'ICIR',o.mean()/o.std(ddof=1),'hit',(o>0).mean(),'turnover',ranks.diff().abs().mean(axis=1).mean())
 for h in [5,10]:
  vals=[]
  for s,x in D.items():
   r=x.close.pct_change();v=r.rolling(20,min_periods=10).std();raw=-r.rolling(3,min_periods=3).sum()/v;f=raw.where(np.arange(len(x))%k==0).ffill()
   vals.append(pd.DataFrame({'date':x.index,'f':f,'y':x.close.shift(-h)/x.close-1}).reset_index(drop=True))
  b=pd.concat(vals,ignore_index=True).dropna();q=[]
  for dt,g in b.groupby('date'):
   if len(g)>=8 and g.f.nunique()>1:q.append(spearmanr(g.f,g.y).statistic)
  print(' decay',h,'IC',np.mean(q),'n',len(q))
