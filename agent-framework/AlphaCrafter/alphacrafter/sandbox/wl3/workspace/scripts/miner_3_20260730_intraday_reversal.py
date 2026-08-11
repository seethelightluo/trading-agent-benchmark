import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def L(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); return d[d.date<='2026-07-15'].set_index('date')
rows=[]
for s in U:
 d=L(s); f=-(d.close/d.open-1); y=d.close.pct_change().shift(-1)
 q=pd.concat([f,y],axis=1); q.columns=['f','y']; q['date']=q.index; q['s']=s; rows.append(q.reset_index(drop=True))
a=pd.concat(rows,ignore_index=True).dropna(); obs=[]
for dt,g in a.groupby('date'):
 if len(g)>=8: obs.append((dt,spearmanr(g.f,g.y).statistic,len(g)))
o=pd.DataFrame(obs,columns=['date','ic','n']).dropna();print('dates',len(o),'avg_n',o.n.mean(),'unique_dates',a.date.nunique(),'coverage',len(o)/a.date.nunique(),'IC',o.ic.mean(),'ICIR',o.ic.mean()/o.ic.std(),'hit',(o.ic>0).mean(),'std',o.ic.std())
for h in [5,10]:
 rows=[]
 for s in U:
  d=L(s); f=-(d.close/d.open-1); y=d.close.pct_change(h).shift(-h); q=pd.concat([f,y],axis=1);q.columns=['f','y'];q['date']=q.index;rows.append(q.reset_index(drop=True))
 b=pd.concat(rows,ignore_index=True).dropna(); z=[]
 for dt,g in b.groupby('date'):
  if len(g)>=8:z.append(spearmanr(g.f,g.y).statistic)
 z=pd.Series(z).dropna();print(h,'IC',z.mean(),'ICIR',z.mean()/z.std(),'obs',len(z))
r=a.copy();r['rank']=r.groupby('date').f.rank(pct=True);r=r.sort_values(['s','date']);print('turnover',r.groupby('s')['rank'].diff().abs().mean())
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 x=o[(o.date.dt.year>=lo)&(o.date.dt.year<=hi)].ic;print('regime',lo,hi,x.mean(),x.mean()/x.std(),len(x))
