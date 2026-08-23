import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date') for a in A}
P=pd.DataFrame({a:d.close.astype(float) for a,d in D.items()}).sort_index().loc[:'2035-07-18']
R=P.pct_change()
# Candidate: directional trend weighted by path efficiency. 20d net return divided by
# total absolute daily movement, then scaled by inverse 20d volatility; lag one day.
net=P.pct_change(20)
path=R.abs().rolling(20,min_periods=15).sum()
vol=R.rolling(20,min_periods=15).std()
F=(net/path/vol).shift(1)
Y=P.shift(-10)/P-1
rows=[]
for dt in F.index:
 q=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)>=8:
  z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
  if np.isfinite(z): rows.append((dt,len(q),z))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('period',r.date.min().date(),r.date.max().date(),'dates',len(r),'avgN',round(r.n.mean(),2),'assets',len(A))
print('full IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(min(k,len(s))); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for h in [1,5,10,20]:
 yy=P.shift(-h)/P-1; rr=[]
 for dt in F.index:
  q=pd.concat([F.loc[dt],yy.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:
   z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(z): rr.append(z)
 print('decay',h,'IC',round(np.mean(rr),6),'ICIR',round(np.mean(rr)/np.std(rr,ddof=1),6))
print('coverage',round(F.notna().sum(axis=1).mean()/len(A),4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
os.makedirs('scripts/artifacts',exist_ok=True)
F.to_csv('scripts/artifacts/miner_2_20350719_efficiency_momentum_20d_signal.csv',index_label='date')
r.to_csv('scripts/artifacts/miner_2_20350719_efficiency_momentum_20d_ic.csv',index=False)
