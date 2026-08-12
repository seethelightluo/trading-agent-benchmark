import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-07-29'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').sort_index() for s in U}
def f(x): return (x.close/x.close.shift(60)-1)/(x.close.pct_change().rolling(20).std()*np.sqrt(20)+1e-12)
def run(h):
 a=pd.concat([pd.DataFrame({'date':x.index,'f':f(x).values,'y':(x.close.shift(-h)/x.close-1).values}) for x in D.values()],ignore_index=True).dropna(); out=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8: out.append((dt,g.f.corr(g.y,method='spearman'),len(g)))
 return pd.DataFrame(out,columns=['date','ic','n']).dropna().set_index('date')
z=run(1); print('60d risk-adjusted momentum; dates',len(z),'avg_n',z.n.mean(),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean())
for h in [5,10,20]:
 q=run(h); print('decay',h,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-29')]:
 q=z.loc[lo:hi].ic; print('regime',lo,len(q),q.mean(),q.mean()/q.std(ddof=1))
r=pd.concat([f(x).rename(s) for s,x in D.items()],axis=1).rank(axis=1,pct=True); print('turnover',r.diff().abs().mean(axis=1).mean(),'coverage',sum(f(x).notna().sum() for x in D.values())/sum(len(x) for x in D.values()))
