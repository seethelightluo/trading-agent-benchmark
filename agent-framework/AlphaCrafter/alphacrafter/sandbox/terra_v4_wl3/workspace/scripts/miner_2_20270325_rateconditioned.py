import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); files=glob.glob('../persistent/stock_data/*.csv')
C={}
for p in files:
 a=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index(); C[a]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); r=close.pct_change();
# Yield-change conditioned residual reversal. Yield shocks are additive percent
# changes in the two tradable yield series; no division by yield levels.
ys=[x for x in ['US10Y','CN10Y'] if x in close]
shock=r[ys].rolling(3).mean().mean(axis=1).abs()
scale=shock.rolling(60,min_periods=30).rank(pct=True)
gate=(scale>=.6).astype(float)
rel=close.pct_change(3).sub(close.pct_change(3).median(axis=1),axis=0)
z=rel.sub(rel.mean(axis=1),axis=0).div(rel.std(axis=1).replace(0,np.nan),axis=0)
fac=(-z).mul(gate,axis=0); fac.to_csv('scripts/miner_2_20270325_rateconditioned_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h); out=[]; ns=[]; dates=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1: out.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic);ns.append(len(x));dates.append(dt)
 s=pd.Series(out,index=dates); return s,ns
print('assets',len(close.columns),'yield_cols',ys,'rows',len(fac),'shock_valid',shock.notna().sum(),'active',int(gate.sum()))
for h in [1,5,10]:
 s,n=ev(h); print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC %.7f ICIR %.7f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,len(q),'IC %.7f ICIR %.7f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(close.columns),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
