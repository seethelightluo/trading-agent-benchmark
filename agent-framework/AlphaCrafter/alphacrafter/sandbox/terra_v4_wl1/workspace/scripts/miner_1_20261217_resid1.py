import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,2500); x=x.copy(); x['date']=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); m=r.median(axis=1); beta=r.rolling(20).cov(m).div(m.rolling(20).var(),axis=0); res=r-beta.mul(m,axis=0)
# 1-day residual reversal, volatility scaled; smooth denominator avoids unstable crypto shocks
f=-res.div(res.rolling(20).std().clip(lower=0.002))
rows=[]
for d in f.index:
 q=pd.concat([f.loc[d],p.pct_change().shift(-1).loc[d]],axis=1).dropna()
 if len(q)>=8: rows.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
a=pd.Series(rows).dropna(); print('daily',a.mean(),a.mean()/a.std(ddof=1),(a>0).mean(),len(a))
for h in [5,10]:
 y=p.pct_change(h).shift(-h+1); z=[]
 for d in f.index:
  q=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 x=pd.Series(z).dropna();print(h,x.mean(),x.mean()/x.std(ddof=1),len(x))
print('coverage',len(a)/(len(f)))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20261217_resid1_signal.csv',index=False)
