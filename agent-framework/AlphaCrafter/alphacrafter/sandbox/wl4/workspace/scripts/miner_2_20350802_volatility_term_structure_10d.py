import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}
P=pd.DataFrame(P).sort_index().loc[:'2035-07-18']; R=P.pct_change()
v5=R.rolling(5,min_periods=4).std(); v30=R.rolling(30,min_periods=15).std()
# positive signal means compressed recent volatility relative to medium term
F=(-(v5/(v30+1e-8))).shift(1).replace([np.inf,-np.inf],np.nan)
rows=[]
for dt in F.index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1; q=pd.concat([F.loc[dt],y],axis=1).dropna()
 if len(q)>=8:
  z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
  if np.isfinite(z): rows.append((dt,len(q),z))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('dates',len(r),'avgN',round(r.n.mean(),2),'coverage',round(r.n.sum()/(len(r)*15),4))
print('full',*[round(x,6) for x in [s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()]])
for k in [120,260,520,780]:
 q=s.tail(k);print('recent',k,round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),round((q>0).mean(),4))
print('decay',end=' ')
for h in [1,5,10,20]:
 z=[]
 for dt in F.index:
  y=P.shift(-h).loc[dt]/P.loc[dt]-1;q=pd.concat([F.loc[dt],y],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 print(h,round(np.nanmean(z),6),end=';')
print('\nturnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
os.makedirs('scripts/artifacts',exist_ok=True);F.to_csv('scripts/artifacts/miner_2_20350802_volatility_term_structure_10d_signal.csv',index_label='date');r.to_csv('scripts/artifacts/miner_2_20350802_volatility_term_structure_10d_ic.csv',index=False)
