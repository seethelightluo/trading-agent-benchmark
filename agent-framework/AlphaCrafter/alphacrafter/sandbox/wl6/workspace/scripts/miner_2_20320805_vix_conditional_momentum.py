import pandas as pd,numpy as np
from scipy.stats import spearmanr
CUT=pd.Timestamp('2032-08-04'); A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in A}).sort_index().loc[:CUT]
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].sort_index().reindex(P.index).ffill()
r=P.pct_change(); mom=P/P.shift(20)-1; down=np.sqrt((r.clip(upper=0)**2).rolling(40,min_periods=25).mean())*np.sqrt(20)
state=(vix>vix.rolling(252,min_periods=100).median()).astype(float)
f=mom.div(down+1e-12)*(1-2*state.to_numpy()[:,None])
f.index=P.index; f.columns=P.columns
print('cutoff',CUT.date(),'rows',len(P),'assets',len(A),'coverage',round(f.notna().stack().mean(),6),'macro_obs',vix.notna().mean())
for h in [5,10,20]:
 xs=[]; ns=[]; ds=[]
 for i in range(len(P)-h):
  z=pd.concat([f.iloc[i].rename('x'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q):xs.append(q);ns.append(len(z));ds.append(P.index[i])
 x=np.array(xs); print('horizon',h,'valid_dates',len(x),'avg_n',round(np.mean(ns),3),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 print('regimes',{int(y):round(float(np.mean([q for q,d in zip(x,ds) if d.year==y])),6) for y in sorted(set(d.year for d in ds))})
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
