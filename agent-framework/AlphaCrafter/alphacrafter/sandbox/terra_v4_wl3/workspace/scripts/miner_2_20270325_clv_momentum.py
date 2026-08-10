import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=2600)
 if d is not None and len(d): d=d.copy(); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date')
p=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index(); hi=pd.DataFrame({s:d.high for s,d in D.items()}).reindex(p.index); lo=pd.DataFrame({s:d.low for s,d in D.items()}).reindex(p.index); r=p.pct_change(); rng=(hi-lo).replace(0,np.nan)
clv=((p-lo)/rng).clip(0,1); mom=p.pct_change(10); vol=r.rolling(20).std(); fac=(mom*(clv.rolling(5,min_periods=3).mean()-.5))/(vol+1e-8)
rows=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],p.pct_change().shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); s=q.ic
print('assets',len(D),'dates',len(s),'avgN',q.n.mean(),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1),'hit',(s>0).mean())
for h in [5,10]:
 vals=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],p.pct_change(h).shift(-h).loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('h',h,'dates',len(vals),'IC',np.mean(vals),'ICIR',np.mean(vals)/np.std(vals,ddof=1))
print('coverage',fac.notna().sum(axis=1).mean()/len(U),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
fac.to_csv('scripts/miner_2_20270325_clv_momentum_signal.csv')
