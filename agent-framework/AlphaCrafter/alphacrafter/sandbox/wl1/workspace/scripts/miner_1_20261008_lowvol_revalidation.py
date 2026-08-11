import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-10-07'); D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
 D[s]=x[x.close.notna()]
# One interpretable idea: inverse 20-day realized volatility, using only trailing closes.
rows=[]
for s,x in D.items():
 r=x.close.pct_change()
 for j in range(30,len(x)-10):
  f=-r.iloc[j-19:j+1].std()
  y=x.close.iloc[j+10]/x.close.iloc[j]-1
  if np.isfinite(f) and np.isfinite(y): rows.append((x.index[j],s,f,y))
a=pd.DataFrame(rows,columns=['date','symbol','factor','fwd10'])
ics=[]; dates=[]; ns=[]; topchanges=[]; prev=None
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>1 and g.fwd10.nunique()>1:
  ics.append(spearmanr(g.factor,g.fwd10).statistic); dates.append(dt); ns.append(len(g))
  ranks=g.set_index('symbol').factor.rank(pct=True)
  if prev is not None: topchanges.append(len(set(ranks.nlargest(3).index)^set(prev))/6)
  prev=ranks.nlargest(3).index
z=np.asarray(ics); dates=np.asarray(dates)
print('idea=inverse_20d_realized_vol horizon=10')
print('dates',len(z),'avgN',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,6),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round(np.mean(z>0),6),'turnover_top3',round(np.mean(topchanges),6))
print('period',dates.min().date(),dates.max().date())
print('annual',{int(y):round(z[dates.astype("datetime64[Y]").astype(int)+1970==y].mean(),6) for y in sorted(set(pd.DatetimeIndex(dates).year))})
for h in [5,20]:
 rr=[]
 for s,x in D.items():
  r=x.close.pct_change()
  for j in range(30,len(x)-h):
   f=-r.iloc[j-19:j+1].std(); y=x.close.iloc[j+h]/x.close.iloc[j]-1
   if np.isfinite(f) and np.isfinite(y): rr.append((x.index[j],f,y))
 q=pd.DataFrame(rr,columns=['date','f','y']); zz=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: zz.append(spearmanr(g.f,g.y).statistic)
 zz=np.asarray(zz); print('decay',h,round(zz.mean(),6),round(zz.mean()/zz.std(ddof=1),6),len(zz))
