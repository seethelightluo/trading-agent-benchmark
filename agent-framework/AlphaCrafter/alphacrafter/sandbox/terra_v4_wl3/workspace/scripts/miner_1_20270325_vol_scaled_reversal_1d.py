import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); C={}
for p in glob.glob('../persistent/stock_data/*.csv'):
 a=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); C[a]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); r=close.pct_change(); vol=r.rolling(20,min_periods=15).std()
# One-day reversal normalized by trailing volatility; cap extreme standardized moves for robustness.
raw=-r.div(vol.replace(0,np.nan)); fac=raw.clip(-3,3)
# Require a valid volatility estimate, then lightly smooth the signal to reduce churn.
fac=fac.rolling(2,min_periods=1).mean()
fac.to_csv('scripts/miner_1_20270325_vol_scaled_reversal_1d_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h); v=[]; ns=[]; ds=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:
   v.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic); ns.append(len(x)); ds.append(dt)
 return pd.Series(v,index=ds),ns
print('assets',len(C),'rows',len(fac))
for h in [1,5,10]:
 s,n=ev(h); print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC %.7f ICIR %.7f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,len(q),'IC %.7f ICIR %.7f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(C),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
for w in [1,2,3,5]:
 q=fac.rolling(w,min_periods=w).mean(); y=close.pct_change().shift(-1); vals=[]
 for dt in q.index:
  x=pd.concat([q.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1: vals.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic)
 print('smooth',w,'IC',np.mean(vals),'ICIR',np.mean(vals)/np.std(vals,ddof=1))
