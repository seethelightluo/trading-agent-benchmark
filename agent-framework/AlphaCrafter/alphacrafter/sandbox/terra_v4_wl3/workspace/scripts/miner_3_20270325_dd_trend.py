import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
C={}
for p in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date')
 C[s]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index()
# Drawdown-aware medium-term trend: 20d momentum, penalized when price is far below its 60d peak.
r20=close.pct_change(20)
peak=close.rolling(60,min_periods=30).max()
dd=(close/peak-1).clip(upper=0)
# positive trend retained; deep drawdowns are down-ranked, with bounded penalty
fac=r20*(1+dd.clip(lower=-0.5))
fac.to_csv('scripts/miner_3_20270325_dd_trend_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h); out=[]; ns=[]; ix=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:
   out.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic); ns.append(len(x)); ix.append(dt)
 return pd.Series(out,index=ix),ns
print('assets',len(C),'rows',len(fac),'period',fac.index.min().date(),fac.index.max().date())
for h in [1,5,10]:
 s,n=ev(h); print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC %.7f ICIR %.7f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,len(q),'IC %.7f ICIR %.7f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(C),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
