import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24');C={}
for p in glob.glob('../persistent/stock_data/*.csv'):
 a=os.path.basename(p)[:-4];d=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index();C[a]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index();r=close.pct_change();
# Medium-horizon momentum divided by trailing volatility, a simple risk-adjusted trend signal.
fac=r.rolling(20,min_periods=15).sum().div(r.rolling(60,min_periods=30).std().replace(0,np.nan))
fac.to_csv('scripts/miner_1_20270327_risk_adj_momentum_signal.csv')
for h in [1,5,10]:
 y=close.pct_change(h).shift(-h);v=[];ns=[];ds=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
 s=pd.Series(v,index=ds);print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,'dates',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(C),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
