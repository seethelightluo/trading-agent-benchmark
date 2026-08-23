import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}).sort_index().loc[:'2035-07-18']
r=P.pct_change(); vol=r.rolling(20,min_periods=15).std(); mom=P.pct_change(30)
F=(-mom/vol).sub((-mom/vol).median(axis=1),axis=0).shift(1); Y=P.shift(-10)/P-1
rows=[]
for dt in F.index:
 q=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)>=8:
  z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
  if np.isfinite(z): rows.append((dt,len(q),z))
a=pd.DataFrame(rows,columns=['date','n','ic']); s=a.ic
print('period',a.date.min().date(),a.date.max().date(),'dates',len(a),'avgN',round(a.n.mean(),2),'assets',len(A),'coverage',round(a.n.mean()/15,4))
print('full',*[round(x,6) for x in [s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()]])
for k in [120,260,520,780]:
 q=s.tail(min(k,len(s))); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for h in [1,5,10,20]:
 y=P.shift(-h)/P-1; z=[]
 for dt in F.index:
  q=pd.concat([F.loc[dt],y.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): z.append(v)
 z=np.array(z); print('decay',h,'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
os.makedirs('scripts/artifacts',exist_ok=True); F.to_csv('scripts/artifacts/miner_2_20350719_crossasset_reversal_30d_signal.csv',index_label='date'); a.to_csv('scripts/artifacts/miner_2_20350719_crossasset_reversal_30d_ic.csv',index=False)
