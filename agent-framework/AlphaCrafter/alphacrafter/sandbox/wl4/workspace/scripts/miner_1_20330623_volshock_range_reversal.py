import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; B='../persistent/stock_data'; F={};P={}
for a in A:
 f=f'{B}/{a}.csv'
 if not os.path.exists(f):continue
 d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date'); c=d.close.astype(float);r=c.pct_change(); P[a]=c
 m=c.pct_change(20); v=r.rolling(60,min_periods=30).std(); vfast=r.rolling(10,min_periods=8).std(); shock=(vfast/v.rolling(60,min_periods=30).mean()).clip(.5,3)
 hi=c.rolling(60,min_periods=40).max();lo=c.rolling(60,min_periods=40).min();loc=((c-lo)/(hi-lo).replace(0,np.nan)).rolling(10,min_periods=5).mean()
 F[a]=(-m/(v*np.sqrt(20))*(.5+loc)*shock).shift(1)
F=pd.DataFrame(F);P=pd.DataFrame(P); rows=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],P.shift(-10).loc[dt]/P.loc[dt]-1],axis=1).dropna()
 if len(z)>=8:rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']);s=r.ic
print('candidate volatility_shock_range_reversal_10d');print('dates',len(r),'avgN',round(r.n.mean(),3),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),6))
for n in [260,520,780]:
 q=s.tail(min(n,len(s)));print('recent',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),6))
print('coverage',round(F.notna().sum(axis=1).mean()/len(A),6),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
os.makedirs('scripts/artifacts',exist_ok=True);r.to_csv('scripts/artifacts/miner_1_20330623_volshock_range_reversal_ic.csv',index=False);F.to_csv('scripts/artifacts/miner_1_20330623_volshock_range_reversal_signal.csv')
