import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv')
 d['date']=pd.to_datetime(d['date']); px[s]=d.set_index('date')['close'].sort_index()
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); m=R['SPX']
# residual momentum: 20d return net of rolling 60d beta to SPX, all lagged by one day
out=[]
for s in U:
 x=R[s]
 cov=x.rolling(60,min_periods=45).cov(m)
 var=m.rolling(60,min_periods=45).var()
 beta=cov/var.replace(0,np.nan)
 f=(P[s].pct_change(20)-beta.shift(1)*P['SPX'].pct_change(20)).shift(0)
 # all features at t predict t+1; beta uses through t, okay
 z=pd.DataFrame({'f':f,'y':R[s].shift(-1)})
 out.append(z.assign(asset=s))
A=pd.concat(out).reset_index().rename(columns={'index':'date'}).dropna()
ics=[]; ranks=[]
for dt,g in A.groupby('date'):
 if len(g)>=8:
  ic=spearmanr(g.f,g.y).statistic
  if np.isfinite(ic): ics.append((dt,ic)); ranks.append(g.assign(r=g.f.rank(pct=True)))
i=pd.Series(dict(ics)).sort_index(); print('dates',len(i),'mean_n',A.groupby('date').size().loc[i.index].mean(),'IC %.5f ICIR %.5f hit %.3f'%(i.mean(),i.mean()/i.std(ddof=1), (i>0).mean()))
# horizons using forward cumulative returns
for h in [1,5,10]:
 rows=[]
 for s in U:
  y=(P[s].shift(-h)/P[s]-1)
  # recompute same f
  x=R[s]; b=x.rolling(60,min_periods=45).cov(m)/m.rolling(60,min_periods=45).var().replace(0,np.nan)
  f=P[s].pct_change(20)-b.shift(1)*P['SPX'].pct_change(20)
  rows.append(pd.DataFrame({'f':f,'y':y,'asset':s}))
 B=pd.concat(rows).reset_index().rename(columns={'index':'date'}).dropna(); ii=[]
 for dt,g in B.groupby('date'):
  if len(g)>=8: ii.append(spearmanr(g.f,g.y).statistic)
 ii=pd.Series(ii).dropna(); print('h',h,'n',len(ii),'IC %.5f ICIR %.5f'%(ii.mean(),ii.mean()/ii.std(ddof=1)))
print('regimes')
for yr in range(2020,2027):
 q=i[i.index.year==yr]
 if len(q): print(yr,len(q),round(q.mean(),4),round(q.mean()/q.std(ddof=1),4))
# approximate turnover of cross-sectional ranks
print('coverage',len(A)/ (len(P)*15),'rank_turnover',A.groupby('date').size().mean())
