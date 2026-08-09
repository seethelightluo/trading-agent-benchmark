import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index() for a in assets}
# Candidate: volatility-normalized medium trend with range-quality gate (one interpretable idea)
# signal = 10d log return / 20d mean absolute return, multiplied by 20d close-location efficiency
all_dates=sorted(set.intersection(*[set(x.index) for x in D.values()]))
rows=[]
for dt in all_dates:
  vals=[]; fwd=[]
  for a in assets:
    x=D[a]; loc=x.index.get_loc(dt)
    if loc<25 or loc+1>=len(x): continue
    c=x.close
    r=np.log(c.iloc[loc]/c.iloc[loc-10])
    ar=np.abs(np.log(c.iloc[loc-20:loc+1].values[1:]/c.iloc[loc-20:loc].values))
    # efficiency: net move / total path, bounded, uses through dt only
    path=np.sum(ar)
    eff=abs(np.log(c.iloc[loc]/c.iloc[loc-20]))/(path+1e-9)
    sig=r/(np.mean(ar)+1e-9)*eff
    # lag signal one day concept: current dt signal predicts next day, valid data through dt
    ret=np.log(c.iloc[loc+1]/c.iloc[loc])
    if np.isfinite(sig) and np.isfinite(ret): vals.append(sig); fwd.append(ret)
  if len(vals)>=8:
    rows.append((dt,spearmanr(vals,fwd).statistic,len(vals)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate=efficiency_gated_volnorm_10d; dates',len(r),'meanN',r.n.mean())
for h in [1]:
 print('IC %.6f ICIR %.6f hit %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(),(r.ic>0).mean()))
for lo,hi in [('2020','2026-12-31'),('2027','2029-12-31'),('2030','2034-12-31')]:
 q=r.loc[lo:hi]; print(lo,'-',hi,len(q),q.ic.mean(),q.ic.mean()/q.ic.std())
# 5d/10d decay using recompute forward aligned
for h in [5,10,20]:
 out=[]
 for dt in r.index:
  vals=[]; fw=[]
  for a in assets:
   x=D[a]; loc=x.index.get_loc(dt)
   if loc+h>=len(x) or loc<25: continue
   c=x.close; ar=np.abs(np.log(c.iloc[loc-20:loc+1].values[1:]/c.iloc[loc-20:loc].values)); sig=np.log(c.iloc[loc]/c.iloc[loc-10])/(np.mean(ar)+1e-9)*(abs(np.log(c.iloc[loc]/c.iloc[loc-20]))/(np.sum(ar)+1e-9))
   vals.append(sig);fw.append(np.log(c.iloc[loc+h]/c.iloc[loc]))
  if len(vals)>=8: out.append(spearmanr(vals,fw).statistic)
 z=np.array(out);print('decay',h,len(z),z.mean(),z.mean()/z.std())
print('coverage cells',len(r)*15,'valid',r.n.sum(),'turnover proxy unavailable')
