import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=sorted(os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv'))
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); px[a]=d.close[d.index<=cut]
prices=pd.DataFrame(px).sort_index(); ret=prices.pct_change()
v20=ret.rolling(20,min_periods=15).std(); v60=ret.rolling(60,min_periods=40).std()
# Volatility compression: low recent volatility relative to its medium-term baseline.
fac=-(v20/(v60+1e-8)-1).replace([np.inf,-np.inf],np.nan).clip(-5,5)
fac.to_csv('scripts/miner_1_20270325_vol_compression_signal.csv')
print('assets',len(assets),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in [1,5,10]:
 fwd=prices.pct_change(h).shift(-h); vals=[];ns=[];ds=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,hi,'IC %.6f ICIR %.6f n %d'%(q.mean(),q.mean()/q.std(ddof=1),len(q)))
print('coverage',fac.notna().sum(axis=1).mean()/len(assets),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
