import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
prices={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').loc[:cut].close for a in assets}
px=pd.DataFrame(prices).sort_index(); ret=px.pct_change(); r3=px.pct_change(3)
peer=r3.mean(axis=1).shift(1); resid=r3.sub(peer,axis=0)
# Contrarian peer-residual move scaled by each asset's lagged 20d volatility.
vol=ret.rolling(20).std().shift(1).replace(0,np.nan)
fac=(-resid/vol).replace([np.inf,-np.inf],np.nan)
fac.to_csv('scripts/miner_3_20270325_scaled_peer_reversal_signal.csv')
print('assets',len(assets),'rows',len(fac),'coverage',fac.notna().sum(axis=1).mean()/len(assets))
for h in [1,5,10]:
 fwd=px.pct_change(h).shift(-h); vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'n',len(q))
print('turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
print('coverage_dates',len(vals))
