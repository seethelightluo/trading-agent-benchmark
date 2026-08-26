import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2035-04-29')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
common=sorted(set.intersection(*[set(d.index) for d in D.values()])); out=[]; dates=[]
for dt in common:
 f={}; fw={}
 for s,d in D.items():
  if dt not in d.index: continue
  i=d.index.get_loc(dt)
  if i<2 or i+1>=len(d): continue
  # range-normalized intraday selloff: reversal strongest when close below open, scaled by prior 20d vol
  x=(d.iloc[i].close/d.iloc[i].open-1)
  vol=d['close'].pct_change().iloc[max(0,i-20):i].std()
  f[s]=-x/max(vol,1e-5); fw[s]=d.iloc[i+1].close/d.iloc[i].close-1
 if len(f)>=8:
  z=spearmanr(list(f.values()),list(fw.values())).statistic
  if np.isfinite(z):out.append(z); dates.append(dt)
x=np.array(out); print('daily intraday reversal dates',len(x),'avg instruments',len(U),'IC %.6f ICIR %.6f hit %.4f'%(x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0)))
for a,b in [('2020-01-01','2026-12-31'),('2027-01-01','2030-12-31'),('2031-01-01','2034-12-31'),('2035-01-01','2035-04-29')]:
 z=x[(np.array(dates)>=pd.Timestamp(a))&(np.array(dates)<=pd.Timestamp(b))]; print(a[:4],len(z),z.mean(),z.mean()/z.std(ddof=1) if len(z)>1 else None)
pd.DataFrame({'date':dates,'ic':x}).to_csv('scripts/miner_3_20350430_intraday_reversal_validation.csv',index=False)
