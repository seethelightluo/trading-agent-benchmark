import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2026-09-23'); D={}
for s in U:
    try:
        x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
        D[s]=x.loc[x.index<=cutoff,'close']
    except Exception as e: print('missing',s,e)
p=pd.DataFrame(D).sort_index(); r=p.pct_change(); market=p.median(axis=1)
# Independent residual momentum: trailing 20-session return unexplained by rolling 60-session beta to cross-asset median.
rows=[]
for s in D:
    beta=r[s].rolling(60,min_periods=45).cov(market)/market.rolling(60,min_periods=45).var()
    f=(p[s]/p[s].shift(20)-1)-beta*(market.rolling(20,min_periods=20).apply(lambda z: np.prod(1+z)-1,raw=True))
    y=p[s].shift(-1)/p[s]-1
    for dt in p.index:
        if pd.notna(f.get(dt)) and pd.notna(y.get(dt)): rows.append((dt,s,float(f.loc[dt]),float(y.loc[dt])))
a=pd.DataFrame(rows,columns=['date','symbol','factor','forward'])
def calc(q):
    z=[]; ns=[]
    for dt,g in q.groupby('date'):
        if len(g)>=8 and g.factor.nunique()>1 and g.forward.nunique()>1:
            z.append(spearmanr(g.factor,g.forward).statistic); ns.append(len(g))
    z=np.asarray(z); return z,ns
z,ns=calc(a)
print('dates',len(z),'avg_names',round(np.mean(ns),3),'symbols',a.symbol.nunique(),'coverage',a.symbol.nunique()/15)
print('daily IC %.8f ICIR %.8f hit %.4f std %.8f'%(z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0),z.std(ddof=1)))
for h in [5,10]:
 q=[]
 for s in D:
  beta=r[s].rolling(60,min_periods=45).cov(market)/market.rolling(60,min_periods=45).var()
  f=(p[s]/p[s].shift(20)-1)-beta*(market.rolling(20,min_periods=20).apply(lambda z: np.prod(1+z)-1,raw=True)); y=p[s].shift(-h)/p[s]-1
  q += [(dt,s,float(f.loc[dt]),float(y.loc[dt])) for dt in p.index if pd.notna(f.get(dt)) and pd.notna(y.get(dt))]
 v,_=calc(pd.DataFrame(q,columns=['date','symbol','factor','forward'])); print('%dd dates %d IC %.8f ICIR %.8f'%(h,len(v),v.mean(),v.mean()/v.std(ddof=1)))
r=a.assign(rank=a.groupby('date').factor.rank(pct=True)).pivot(index='date',columns='symbol',values='rank').sort_index()
print('rank_turnover',r.diff().abs().mean().mean())
for label,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-09-23')]:
 sub=a[(a.date>=lo)&(a.date<=hi)]; v,_=calc(sub); print(label,'dates',len(v),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1))
