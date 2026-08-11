import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
CUT=pd.Timestamp('2027-04-21')
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).query('date<=@CUT').set_index('date') for s in U}
def factor(d):
 r=d.close.pct_change(); up=r.clip(lower=0).rolling(20).sum(); down=(-r.clip(upper=0)).rolling(20).sum(); vol=r.rolling(20).std(); return (up/(down+.01))/(vol+.005)
def eval_h(h):
 rows=[]
 for s,d in D.items(): rows.append(pd.DataFrame({'f':factor(d).shift(1),'y':d.close.pct_change(h).shift(-h),'date':d.index,'s':s}).dropna())
 x=pd.concat(rows,ignore_index=True); ic=[]; ns=[]
 for dt,g in x.groupby('date'):
  g=g.replace([np.inf,-np.inf],np.nan).dropna()
  if len(g)>=8:ic.append(spearmanr(g.f,g.y).statistic);ns.append(len(g))
 a=np.array(ic); return a,np.mean(ns),len(x)/sum(len(d) for d in D.values()),x
for h in [1,3,5,10]:
 a,n,c,x=eval_h(h); print('h',h,'dates',len(a),'avgN',n,'coverage',c,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
a,n,c,x=eval_h(1); ranks=x.assign(rank=x.groupby('date').f.rank(pct=True)).pivot(index='date',columns='s',values='rank').sort_index();print('turnover',ranks.diff().abs().mean(axis=1).mean())
for lo,hi in [('2020','2021'),('2022','2023'),('2024','2025'),('2026','2027')]:
 q=[v for dt,v in zip(sorted(x.date.unique()),a) if lo<=str(dt)[:4]<=hi];print(lo+'-'+hi,'n',len(q),'IC',np.mean(q) if q else np.nan)
print('last',x.date.max())
