import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); A=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
P={}; R={h:{} for h in [1,5,10]}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').sort_index(); r=d.close.pct_change();
 # Short-term contrarian impulse, with volatility cap and cross-sectional demeaning.
 P[a]=(-r.rolling(3,min_periods=3).sum()/r.rolling(20,min_periods=10).std()).clip(-10,10)
 for h in R:R[h][a]=d.close.pct_change(h).shift(-h)
f=pd.DataFrame(P).sort_index(); f.to_csv('scripts/miner_1_20270325_short_reversal3_signal.csv')
print('assets',len(A),'rows',len(f),'period',f.index.min(),f.index.max())
for h in R:
 q=pd.DataFrame(R[h]).reindex(f.index); x=[]; ds=[]; ns=[]
 for t in f.index:
  z=pd.concat([f.loc[t],q.loc[t]],axis=1).dropna()
  if len(z)>=8 and z.nunique().min()>1:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(t);ns.append(len(z))
 s=pd.Series(x,index=ds);print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f cov %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),f.notna().sum(axis=1).mean()/len(A)))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   z=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,hi,'IC %.6f ICIR %.6f n %d'%(z.mean(),z.mean()/z.std(ddof=1),len(z)))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
