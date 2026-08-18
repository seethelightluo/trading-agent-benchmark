import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); P[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(P).sort_index(); R=P.pct_change(); m=R.mean(axis=1)
# market beta adjusted by idiosyncratic volatility: beta / residual vol; lower is defensive
f=pd.DataFrame(index=P.index,columns=P.columns,dtype=float)
for s in U:
 cov=R[s].rolling(60,min_periods=40).cov(m); var=m.rolling(60,min_periods=40).var()
 beta=cov/var
 resid=R[s]-beta*m
 rv=resid.rolling(40,min_periods=25).std()
 f[s]=-(beta/(rv+1e-8))
fwd=P.shift(-10)/P-1
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate=defensive_beta_residual_60_40'); print('dates',len(r),'mean_n',r.n.mean(),'coverage',r.n.sum()/(len(r)*15)); print('IC %.8f ICIR %.8f hit %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1),(r.ic>0).mean()))
for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2027','2028')]:
 x=r.loc[a:b]
 if len(x): print(a+'-'+b,len(x),'IC %.8f ICIR %.8f'%(x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1)))
print('rank_change',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [1,5,10,20]:
 rr=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],(P.shift(-h)/P-1).loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('horizon',h,'IC',np.mean(rr),'dates',len(rr))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20280114_defensive_beta_signal.csv',index=False)
