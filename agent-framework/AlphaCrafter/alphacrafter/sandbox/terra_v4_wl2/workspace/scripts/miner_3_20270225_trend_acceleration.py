import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
r=pd.DataFrame({a:p[a].pct_change() for a in A})
# Trend acceleration: recent 5d return relative to prior 20d return, normalized by trailing 20d volatility. Lagged one day.
f=((r.rolling(5,min_periods=5).sum()-r.rolling(25,min_periods=25).sum().shift(5))/(r.rolling(20,min_periods=15).std()+1e-8)).shift(1)
# forward close-close returns
outs=[]
for h in [1,5,10]:
 vals=[]
 for d in f.index:
  z=[]
  for a in A:
   if d not in p[a].index: continue
   i=p[a].index.get_loc(d)
   if i+h>=len(p[a]): continue
   if np.isfinite(f.loc[d,a]): z.append((f.loc[d,a],p[a].iloc[i+h]/p[a].iloc[i]-1))
  if len(z)>=8:
   x,y=zip(*z); vals.append((d,spearmanr(x,y).statistic,len(z)))
 q=pd.DataFrame(vals,columns=['date','ic','n']);outs.append(q)
 print('H',h,'dates',len(q),'avg_n',q.n.mean(),'coverage',len(q)/len(f),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean())
 for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-06-30'),('2026-07','2027-02-24')]:
  x=q.set_index('date').loc[lo:hi].ic; print(' regime',lo,len(x),x.mean() if len(x) else np.nan,(x.mean()/x.std(ddof=1)) if len(x)>1 else np.nan)
q=f.rank(axis=1,pct=True);print('turnover',np.nanmean((q-q.shift()).abs().mean(axis=1)))
f.stack().rename('signal').reset_index().rename(columns={'level_1':'symbol'}).to_csv('../persistent/factor_signals_miner_3_20270225_trend_acceleration.csv',index=False)
