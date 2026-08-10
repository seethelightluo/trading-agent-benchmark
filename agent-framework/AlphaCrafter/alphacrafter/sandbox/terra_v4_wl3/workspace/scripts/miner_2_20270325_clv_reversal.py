import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); base='../persistent/stock_data/'
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in U:
 fn=base+a+'.csv'
 if os.path.exists(fn):
  d=pd.read_csv(fn,parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date'); D[a]=d
px=pd.DataFrame({a:d.close for a,d in D.items()}).sort_index();
high=pd.DataFrame({a:d.high for a,d in D.items()}).reindex(px.index); low=pd.DataFrame({a:d.low for a,d in D.items()}).reindex(px.index)
r=px.pct_change(); rng=(high-low).replace(0,np.nan)
# Close-location reversal: fade unusually strong closes, averaged over 3 sessions and scaled by 20d volatility.
clv=((px-low)/rng).clip(0,1)
raw=-(clv-.5)
fac=raw.rolling(3,min_periods=2).mean()/(r.rolling(20,min_periods=15).std()+1e-8)
fac.to_csv('scripts/miner_2_20270325_clv_reversal_signal.csv')
print('assets',len(D),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in [1,5,10]:
 fwd=px.pct_change(h).shift(-h); vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(dt); ns.append(len(z))
 s=pd.Series(vals,index=ds)
 print('H',h,'dates',len(s),'avgN %.2f IC %.7f ICIR %.7f hit %.4f'%(np.mean(ns),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,'IC %.7f ICIR %.7f n %d'%(q.mean(),q.mean()/q.std(ddof=1),len(q)))
print('coverage %.6f turnover %.6f'%(fac.notna().sum(axis=1).mean()/len(U),fac.rank(axis=1,pct=True).diff().abs().mean().mean()))
print('valid_dates',len(s))
