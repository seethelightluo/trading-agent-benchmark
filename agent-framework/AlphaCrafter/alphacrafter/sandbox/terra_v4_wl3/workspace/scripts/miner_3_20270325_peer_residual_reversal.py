import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]; R={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date');d=d[d.index<=cut];R[a]=d.close.pct_change(3)
r=pd.DataFrame(R).sort_index(); peer=r.mean(axis=1).shift(1)
# Contrarian residual short-horizon move vs lagged cross-asset benchmark
fac=-(r.sub(peer,axis=0)); fac.to_csv('scripts/miner_3_20270325_peer_residual_reversal_signal.csv')
print('assets',len(assets),'rows',len(fac),'coverage',fac.notna().sum(axis=1).mean()/len(assets))
for h in [1,5,10]:
 fwd=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.pct_change(h).shift(-h) for a in assets}).reindex(fac.index); vals=[];ds=[];ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=ds);print('H',h,'dates',len(s),'avgN',np.mean(ns),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1),'hit',(s>0).mean())
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'n',len(q))
print('turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
