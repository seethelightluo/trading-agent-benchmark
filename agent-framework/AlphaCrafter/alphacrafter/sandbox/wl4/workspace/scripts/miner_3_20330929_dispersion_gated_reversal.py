import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f): P[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').close.astype(float)
P=pd.DataFrame(P).sort_index().loc[:'2033-09-28']; R=P.pct_change(); vol=R.rolling(20,min_periods=15).std()*np.sqrt(252)
cross=R.apply(lambda x:x.std(),axis=1).rolling(10,min_periods=8).mean(); scale=(cross/cross.rolling(60,min_periods=30).median()).clip(.5,2)
F=(-P.pct_change(10).div(vol)).mul(scale,axis=0).shift(1)
rows=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],(P.shift(-10).loc[dt]/P.loc[dt]-1)],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('period',r.date.min(),r.date.max(),'dates',len(r),'avgN',r.n.mean(),'assets',len(P.columns));print('IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',(s>0).mean())
for k in [120,260,520,780]:
 q=s.tail(k); print('recent',k,'IC',q.mean(),'ICIR',q.mean()/q.std())
print('coverage',F.notna().sum(axis=1).mean()/len(P.columns),'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
os.makedirs('scripts/artifacts',exist_ok=True);r.to_csv('scripts/artifacts/miner_3_20330929_dispersion_gated_reversal_ic.csv',index=False);F.to_csv('scripts/artifacts/miner_3_20330929_dispersion_gated_reversal_signal.csv')
