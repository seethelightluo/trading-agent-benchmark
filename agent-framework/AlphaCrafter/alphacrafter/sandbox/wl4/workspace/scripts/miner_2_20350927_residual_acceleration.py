import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}
P=pd.DataFrame(P).sort_index().loc[:'2035-09-26']; R=P.pct_change(); bench=R.mean(axis=1)
# Residual acceleration: asset 20d return minus contemporaneous equal-weight market return, blended with 5d residual impulse.
r20=P.pct_change(20)-P.shift(20).div(P.shift(20)).mul(0) # placeholder
res=R.sub(bench,axis=0); raw=(0.7*res.rolling(20,min_periods=15).sum()+0.3*res.rolling(5,min_periods=4).sum()).shift(1)
F=raw.sub(raw.mean(axis=1),axis=0).div(raw.std(axis=1).replace(0,np.nan),axis=0)
rows=[]
for dt in F.index:
 q=pd.concat([F.loc[dt],P.shift(-10).loc[dt]/P.loc[dt]-1],axis=1).dropna()
 if len(q)>=8: rows.append((dt,len(q),spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('period',r.date.min().date(),r.date.max().date(),'dates',len(r),'avgN',round(r.n.mean(),2),'assets',len(A))
print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(min(k,len(s))); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('coverage',round(F.notna().sum(axis=1).mean()/len(A),4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
os.makedirs('scripts/artifacts',exist_ok=True); r.to_csv('scripts/artifacts/miner_2_20350927_residual_acceleration_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_2_20350927_residual_acceleration_signal.csv')
