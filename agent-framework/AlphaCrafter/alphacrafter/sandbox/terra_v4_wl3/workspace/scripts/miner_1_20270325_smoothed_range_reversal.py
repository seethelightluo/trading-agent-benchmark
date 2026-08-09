import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
F={}; P={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date')
 d=d[d.date<=cut].set_index('date'); c=d.close.astype(float); r=c.pct_change()
 rng=((d.high-d.low)/c).replace([np.inf,-np.inf],np.nan)
 # Smoothed range-normalized one-day reversal: average lagged daily reversal over 3 sessions,
 # normalized by trailing median range. All inputs are observable at signal date.
 denom=rng.rolling(10,min_periods=6).median()+0.001
 raw=-r/denom
 F[a]=raw.rolling(3,min_periods=2).mean()
 P[a]=c
fac=pd.DataFrame(F).sort_index(); px=pd.DataFrame(P).sort_index()
out='scripts/miner_1_20270325_smoothed_range_reversal_signal.csv'; fac.to_csv(out)
print('assets',len(assets),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in [1,5,10]:
 fwd=px.pct_change(h).shift(-h); vals=[]; ns=[]; dates=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
 s=pd.Series(vals,index=dates)
 print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,'n',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(assets),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
