import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2032-09-19')
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  x=d.copy(); x.date=pd.to_datetime(x.date); x=x[x.date<=cutoff].set_index('date').close.astype(float); px[s]=x
P=pd.DataFrame(px).sort_index().ffill()
r=P.pct_change(); mom=P/P.shift(20)-1
# Path efficiency: directional 20d move divided by total absolute daily path, lagged at t
path=r.abs().rolling(20,min_periods=18).sum()
F=mom/path
# volatility scaling modestly improves comparability without changing sign
F=F.div(r.rolling(40,min_periods=30).std()).replace([np.inf,-np.inf],np.nan)
fr={h:P.shift(-h)/P-1 for h in [1,5,10,20]}
print('cutoff',cutoff.date(),'calendar_dates',len(P),'assets',len(px),'avg_valid',round(F.notna().sum(axis=1).mean(),2))
for h in [1,5,10,20]:
 vals=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],fr[h].loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1])); ns.append(len(z))
 ic=pd.Series(vals); print('H',h,'n',len(ic),'avgN',round(np.mean(ns),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
ranks=F.rank(axis=1,pct=True); print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(ranks.diff().abs().mean(axis=1).mean(),4))
# thirds H10/H20
for h in [10,20]:
 a=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],fr[h].loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 n=len(a)//3; print('thirds',h,[round(np.mean(a[i*n:(i+1)*n]),6) for i in range(3)])
out=F.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_1_20320920_path_efficiency_signal.csv',index=False); print('artifact_rows',len(out))
