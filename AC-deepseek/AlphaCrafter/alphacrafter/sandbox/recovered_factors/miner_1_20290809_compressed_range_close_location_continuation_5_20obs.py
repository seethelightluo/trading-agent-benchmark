"""One-factor validation: compressed-range close-location continuation."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2029-08-08')
def load(a):
 p='../persistent/stock_data/'+a+'.csv'
 return pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index().loc[:END]
d={a:load(a) for a in A}
c=pd.DataFrame({a:x.close for a,x in d.items()}); h=pd.DataFrame({a:x.high for a,x in d.items()}); l=pd.DataFrame({a:x.low for a,x in d.items()})
# Close location is informative chiefly when the completed bar is unusually compressed:
# a directional close following range compression is treated as a breakout-continuation signal.
clv=(2*c-h-l)/(h-l).replace(0,np.nan)
rng=(h-l).div(c.shift(1)).abs()
relative_range=rng.div(rng.rolling(20,min_periods=12).median())
compression=(1-relative_range).clip(-1,1)
f=(clv*compression).rolling(5,min_periods=4).mean()
print('FACTOR compressed_range_close_location_continuation_5_20obs visible_through',END.date(),'assets',len(A),'signal_cells',int(f.notna().sum().sum()),'/',f.size)
def getic(y):
 out=[]; ns=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):out.append((t,v));ns.append(len(q))
 return pd.Series(dict(out)),float(np.mean(ns)) if ns else 0
vals={}
for horizon in [1,5,10,20]:
 s,n=getic(c.shift(-horizon).div(c)-1); vals[horizon]=s
 print('H',horizon,'dates',len(s),'IC %.6f ICIR %.6f hit %.4f mean_instruments %.2f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),n))
# turnover and coverage independently reported
ranks=f.rank(axis=1); to=[]
for i in range(1,len(ranks)):
 q=pd.concat([ranks.iloc[i-1],ranks.iloc[i]],axis=1).dropna()
 if len(q)>=8: to.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('TURNOVER %.6f COVERAGE %.4f'%(np.mean(to),f.notna().to_numpy().mean()))
for horizon,s in vals.items():
 print('DECAY_H',horizon,'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6))
for label,a,b in [('2020-21','2020-01-01','2022-01-01'),('2022-23','2022-01-01','2024-01-01'),('2024-25','2024-01-01','2026-01-01'),('2026-current','2026-01-01','2030-01-01')]:
 s=vals[5][(vals[5].index>=a)&(vals[5].index<b)]
 print('REGIME',label,'horizon 5 dates',len(s),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6) if len(s)>1 else np.nan,'hit',round((s>0).mean(),4) if len(s) else np.nan)
# preserve selected signal for a correlation test only if performance clears gates
best=max(vals, key=lambda k:abs(vals[k].mean())*abs(vals[k].mean()/vals[k].std(ddof=1)))
print('SELECTED_HORIZON',best)
