import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); files=glob.glob('../persistent/stock_data/*.csv')
assets=sorted(os.path.basename(p)[:-4] for p in files); C={}
for p in files:
 d=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index(); C[os.path.basename(p)[:-4]]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); r=close.pct_change(); market=r.median(axis=1)
spread=np.log(close['US10Y'].clip(lower=1e-9))-np.log(close['CN10Y'].clip(lower=1e-9))
shock=spread.diff(5); center=shock.rolling(120,min_periods=60).median(); mad=(shock-center).abs().rolling(120,min_periods=60).median()
z=((shock-center)/(mad+1e-5)).clip(-3,3).fillna(0)
F={}
for a in assets:
 beta=r[a].rolling(60,min_periods=30).cov(market)/market.rolling(60,min_periods=30).var()
 resid=(r[a]-beta*market).fillna(0)
 F[a]=(-resid.rolling(3,min_periods=3).sum()*(1+0.25*z.abs())).clip(-.2,.2)
fac=pd.DataFrame(F); fac.to_csv('scripts/miner_2_20270325_rate_shock_residual_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h); vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1: vals.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic);ds.append(dt);ns.append(len(x))
 return pd.Series(vals,index=ds),ns
print('assets',len(assets),'rows',len(fac))
for h in [1,5,10]:
 s,n=ev(h); print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,'IC %.6f n %d'%(q.mean(),len(q)))
print('coverage',fac.notna().sum(axis=1).mean()/len(assets),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
