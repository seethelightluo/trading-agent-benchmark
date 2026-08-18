import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; S={}; P={}
for a in A:
 f=f'{base}/{a}.csv'
 if not os.path.exists(f): continue
 d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date'); c=d.close.astype(float); r=c.pct_change()
 mom=c.pct_change(5); vol=r.rolling(40,min_periods=20).std()*np.sqrt(5)
 hi=c.rolling(40,min_periods=25).max(); lo=c.rolling(40,min_periods=25).min(); loc=((c-lo)/(hi-lo).replace(0,np.nan)).rolling(5,min_periods=3).mean()
 S[a]=(-mom/vol*(0.5+loc)).shift(1); P[a]=c
F=pd.DataFrame(S); Q=pd.DataFrame(P); out=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],Q.shift(-10).loc[dt]/Q.loc[dt]-1],axis=1).dropna()
 if len(z)>=8: out.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(out,columns=['date','n','ic']); s=r.ic
print('dates',len(r),'avgN',round(r.n.mean(),3),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),6))
for n in [260,520,780]:
 q=s.tail(n); print('recent',n,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6),'hit',round((q>0).mean(),6))
print('coverage',round(F.notna().sum(axis=1).mean()/len(A),6),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
os.makedirs('scripts/artifacts',exist_ok=True); r.to_csv('scripts/artifacts/miner_3_20330526_short_range_reversal_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_3_20330526_short_range_reversal_signal.csv')
