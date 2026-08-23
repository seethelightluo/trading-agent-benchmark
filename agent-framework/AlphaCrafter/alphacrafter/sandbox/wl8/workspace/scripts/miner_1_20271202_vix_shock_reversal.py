import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];END=pd.Timestamp('2027-12-01')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv');x.date=pd.to_datetime(x.date);P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index();
v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date);v=v[v.date<=END].set_index('date').close.sort_index().reindex(px.index).ffill()
shock=(v.pct_change(3).shift(1)>v.pct_change(3).shift(1).rolling(120).quantile(.8)).astype(float)
sig=-(px.pct_change(2).shift(1)).mul(shock,axis=0)
fwd=px.shift(-1)/px-1; rows=[]
for d in px.index:
 g=pd.DataFrame({'s':sig.loc[d],'f':fwd.loc[d]}).dropna()
 if len(g)>=8 and g.s.nunique()>1:
  z=spearmanr(g.s,g.f).statistic
  if np.isfinite(z):rows.append((d,z,len(g)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def st(q):
 z=a.loc[q,'ic'];return len(z),round(a.loc[q,'n'].mean(),2),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6),round((z>0).mean(),4)
print('dates',len(a),'rows',int(a.n.sum()),'overall',st(slice(None)))
y=a.index.year
for n,q in [('2026',y==2026),('2027',y==2027),('last180',a.index>=END-pd.Timedelta(days=180))]:print(n,st(q))
print('coverage',round(np.isfinite(sig).sum().sum()/sig.size,4))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20271202_vix_shock_reversal_signal.csv',index=False)
