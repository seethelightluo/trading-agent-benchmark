import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2026-12-30')
assets=sorted([os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')])
F={}; FW={h:{} for h in [1,5,10]}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date')
 d=d[d.date<=cut].set_index('date'); r=d.close.pct_change()
 # Two-session contrarian return scaled by trailing 20-session volatility.
 F[a]=(-r.rolling(2,min_periods=2).sum()/r.rolling(20,min_periods=15).std()).replace([np.inf,-np.inf],np.nan)
 for h in FW: FW[h][a]=d.close.pct_change(h).shift(-h)
fac=pd.DataFrame(F).sort_index(); path='scripts/miner_2_20261231_short_reversal_risk_scaled_2d_signal.csv'; fac.to_csv(path)
print('assets',len(assets),'rows',len(fac),'period',fac.index.min(),fac.index.max(),'signal',path)
for h in [1,5,10]:
 fwd=pd.DataFrame(FW[h]).reindex(fac.index); vals=[];ds=[];ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=ds)
 print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print(label,'IC %.6f n %d hit %.4f'%(q.mean(),len(q),(q>0).mean()))
print('coverage',fac.notna().sum(axis=1).mean()/len(assets),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
