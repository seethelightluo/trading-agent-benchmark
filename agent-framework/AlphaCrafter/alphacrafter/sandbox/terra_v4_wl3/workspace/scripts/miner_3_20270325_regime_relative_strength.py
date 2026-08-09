import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]; C={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); C[a]=d[d.date<=cut].set_index('date').close
px=pd.DataFrame(C).sort_index(); ret=px.pct_change(); market=ret['SPX'].rolling(20,min_periods=15).mean()
# regime-gated relative strength: trend direction follows broad market, but bearish regime rewards recent defensive relative strength via cross-sectional reversal
base=ret.rolling(10,min_periods=8).sum(); med=base.median(axis=1); fac=base.sub(med,axis=0).mul(np.where(market>=0,1,-1),axis=0)
fwd=ret.shift(-1); fac.to_csv('scripts/miner_3_20270325_regime_relative_strength_signal.csv')
vals=[];ds=[];ns=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
s=pd.Series(vals,index=ds); print('regime_relative_strength dates',len(s),'avgN',np.mean(ns),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1),'hit',(s>0).mean(),'coverage',fac.notna().sum(axis=1).mean()/15,'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
 q=s[(s.index>=lo)&(s.index<=hi)]; print(lo,len(q),q.mean(),q.mean()/q.std(ddof=1))
for h in [5,10]:
 y=ret.rolling(h).sum().shift(-h); vs=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: vs.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(vs);print(h,q.mean(),q.mean()/q.std(ddof=1))
