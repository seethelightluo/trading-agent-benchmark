import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}
P=pd.DataFrame(P).sort_index().loc[:'2035-10-24']; R=P.pct_change()
down15=R.clip(upper=0).rolling(15,min_periods=11).std()
down20=R.clip(upper=0).rolling(20,min_periods=14).std()
# Candidate: 15d tail-risk reversal, orthogonalized cross-sectionally to the established 20d downside reversal.
x=(-(P.pct_change(15))/down15.replace(0,np.nan)).shift(1)
z=(-(P.pct_change(20))/down20.replace(0,np.nan)).shift(1)
def csstd(v): return v.sub(v.mean(axis=1),axis=0).div(v.std(axis=1).replace(0,np.nan),axis=0)
x=csstd(x); z=csstd(z)
F=pd.DataFrame(index=P.index,columns=A,dtype=float)
for dt in P.index:
 q=pd.concat([x.loc[dt],z.loc[dt]],axis=1).dropna()
 if len(q)>=8:
  xx=q.iloc[:,0].to_numpy(); zz=q.iloc[:,1].to_numpy()
  beta=np.dot(zz,xx)/max(np.dot(zz,zz),1e-12)
  F.loc[dt,q.index]=xx-beta*zz
F=csstd(F)
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
# signal provenance and candidate-vs-library proxy correlation
base=csstd(z)
print('mean_abs_signal_corr_to_20d',round(np.nanmean([F.loc[d].corr(base.loc[d],method='spearman') for d in F.index]),6))
os.makedirs('scripts/artifacts',exist_ok=True)
r.to_csv('scripts/artifacts/miner_3_20351025_residual_tail_reversal_15d_ic.csv',index=False)
F.to_csv('scripts/artifacts/miner_3_20351025_residual_tail_reversal_15d_signal.csv')
