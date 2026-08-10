import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U},axis=1).sort_index().loc[:END]
r=P.pct_change(); ret=r.rolling(4,min_periods=4).sum(); med=ret.median(axis=1); vol=r.rolling(20,min_periods=15).std(); f=(-(ret.sub(med,axis=0))/vol).shift(1)
rows=[]
for s in U: rows.append(pd.DataFrame({'date':P.index,'symbol':s,'factor':f[s],'y':P[s].shift(-1)/P[s]-1,'y5':P[s].shift(-5)/P[s]-1,'y10':P[s].shift(-10)/P[s]-1}))
x=pd.concat(rows,ignore_index=True)
def calc(col):
 a=[]
 for d,g in x.groupby('date'):
  g=g.dropna(subset=['factor',col])
  if len(g)>=8 and g.factor.nunique()>1: 
   z=spearmanr(g.factor,g[col]).statistic
   if np.isfinite(z): a.append((d,z,len(g)))
 a=pd.DataFrame(a,columns=['date','ic','n']); q=a.ic
 return len(a),a.n.mean(),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),a
for c in ['y','y5','y10']:
 z=calc(c); print(c,tuple(z[:5]))
 if c=='y':
  for lo,hi,n in [('2020','2022','20-22'),('2023','2024','23-24'),('2025','2026-12-17','25-26')]:
   q=z[5]; q=q[(q.date>=lo)&(q.date<=hi)].ic; print(n,len(q),q.mean(),q.mean()/q.std(ddof=1))
v=x.dropna(subset=['factor']); rank=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('coverage',len(v)/len(x),'turnover',rank.diff().abs().mean(axis=1).mean())
v[['date','symbol','factor']].to_csv('scripts/miner_1_20261217_volscaled_residual4_signal.csv',index=False)
