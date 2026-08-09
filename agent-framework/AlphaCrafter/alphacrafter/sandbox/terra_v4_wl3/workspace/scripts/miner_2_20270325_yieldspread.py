import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
assets=sorted([os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')])
cl={}; rr={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); d=d[d.index<=cut]
 cl[a]=d.close; rr[a]=d.close.pct_change()
r=pd.DataFrame(rr).sort_index(); prices=pd.DataFrame(cl).sort_index(); market=r.median(axis=1)
# Tradable yield-series spread is used only as a conditioning macro regime; no orders on observation-only symbols.
spread=np.log(prices['US10Y'].clip(lower=1e-9))-np.log(prices['CN10Y'].clip(lower=1e-9))
shock=spread.diff(5); med=shock.rolling(120,min_periods=60).median(); mad=(shock-med).abs().rolling(120,min_periods=60).median()
stress=((shock-med)/(mad+1e-6)).clip(-3,3).reindex(r.index).fillna(0)
F={}
for a in assets:
 beta=r[a].rolling(60,min_periods=30).cov(market)/market.rolling(60,min_periods=30).var()
 resid=r[a]-beta*market
 # Reversal of idiosyncratic daily shock, strengthened under extreme yield-spread moves.
 F[a]=(-resid*(1+0.30*stress.abs())).clip(-.2,.2)
fac=pd.DataFrame(F).sort_index(); fac.to_csv('scripts/miner_2_20270325_yieldspread_residual_signal.csv')
print('assets',len(assets),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in [1,5,10]:
 fwd=prices.pct_change(h).shift(-h); vals=[]; ns=[]; ds=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)]; print('regime',lo,hi,'IC %.6f n %d'%(q.mean(),len(q)))
print('coverage',fac.notna().sum(axis=1).mean()/len(assets),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
