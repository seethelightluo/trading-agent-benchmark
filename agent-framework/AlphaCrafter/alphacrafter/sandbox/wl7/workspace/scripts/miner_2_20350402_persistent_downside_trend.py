import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
xs={}
for s in U:
 d=get_stock_daily_data(s,days=4200)
 if d is not None and len(d)>100: xs[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(xs).sort_index().ffill()
r=p.pct_change()
# candidate: persistent trend: 20d return divided by downside deviation over 40d, with sign-preserving efficiency
mom=p.pct_change(20)
down=r.where(r<0,0).rolling(40,min_periods=20).std()
fac=mom/(down+1e-8)
# forward 10d return
fwd=p.shift(-10)/p-1
ics=[]; dates=[]; cov=[]; turns=[]
prev=None
for dt in fac.index:
 a=fac.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8:
  ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); cov.append(len(z)/len(U))
  if prev is not None: turns.append((a.rank()-prev.rank()).abs().mean()/len(U))
  prev=a
ic=pd.Series(ics,index=dates).dropna()
print('assets',len(xs),'dates',len(ic),'period',ic.index.min(),ic.index.max(),'meanIC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit', (ic>0).mean(),'coverage',np.mean(cov),'turnover',np.mean(turns))
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; vals=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(vals).dropna(); print('H',h,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
for label,(a,b) in {'2020-2024':('2020','2024-12-31'),'2025-2029':('2025','2029-12-31'),'2030-now':('2030','2035-04-02')}.items():
 q=ic.loc[a:b]; print(label,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
# signal artifact for audit
out=fac.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal'); out.to_csv('scripts/miner_2_20350402_persistent_downside_trend_signal.csv',index=False)
