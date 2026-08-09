import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}; p=pd.DataFrame(p).sort_index(); r=p.pct_change(); spread=p['US10Y']-p['CN10Y']; ds=spread.diff()
# conditional rate-shock exposure: assets with historically positive beta to yield spread are penalized during rising spread shocks
fac=pd.DataFrame(index=p.index,columns=U,dtype=float)
for s in U:
 cov=r[s].rolling(60).cov(ds); var=ds.rolling(60).var(); beta=cov/var
 fac[s]=-beta*ds.rolling(5).sum()
rows=[]
for i in range(len(p)-1):
 z=pd.concat([fac.iloc[i],r.iloc[i+1]],axis=1).dropna()
 if len(z)>=8: rows.append((p.index[i],len(z),z.iloc[:,0].corr(z.iloc[:,1])))
d=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); a=d.ic
print('daily',a.mean(),a.std(),a.mean()/a.std(),(a>0).mean(),'dates',len(a),'avgN',d.n.mean())
for h in [5,10]:
 q=[]
 for i in range(len(p)-h):
  z=pd.concat([fac.iloc[i],p.pct_change(h).iloc[i+h]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1]))
 q=pd.Series(q);print(h,q.mean(),q.mean()/q.std(),len(q))
print('coverage',fac.notna().sum(axis=1).mean()/15,'turnover',fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
fac.to_csv('scripts/miner_2_20270325_rate_shock_beta_signal.csv',index_label='date')
