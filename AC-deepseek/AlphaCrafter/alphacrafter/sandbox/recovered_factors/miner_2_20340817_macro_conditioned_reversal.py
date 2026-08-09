import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index() for a in A}
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().close
common=sorted(set.intersection(*[set(x.index) for x in D.values()])&set(v.index))
# Macro-conditioned short reversal: reverse recent 3d return, amplified during a VIX rise.
rows=[]; regimes={}; turnover=[]; prev=None; cov=[]
for k,dt in enumerate(common):
 if k<25 or k+1>=len(common): continue
 j=v.index.get_loc(dt); shock=np.log(v.iloc[j]/v.iloc[j-1]); hist=np.log(v.iloc[j-20:j]/v.iloc[j-21:j]).values
 z=shock/(np.std(hist)+1e-9); amp=1+np.clip(z, -1, 3) # stress amplifies reversal, calm suppresses
 vals=[]; fw=[]
 for a in A:
  x=D[a]; i=x.index.get_loc(dt)
  if i<4 or i+1>=len(x): continue
  r=np.log(x.close.iloc[i]/x.close.iloc[i-3])
  vals.append(-r*amp); fw.append(np.log(x.close.iloc[i+1]/x.close.iloc[i]))
 if len(vals)>=8:
  ic=spearmanr(vals,fw).statistic; rows.append((dt,ic,len(vals),z))
  cov.append(len(vals)/15)
  q=np.argsort(np.argsort(vals)); turnover.append(np.mean(q!=prev) if prev is not None else np.nan); prev=q
r=pd.DataFrame(rows,columns=['date','ic','n','z']).set_index('date')
print('candidate=macro_conditioned_3d_reversal dates',len(r),'meanN',r.n.mean(),'coverage',np.mean(r.n)/15,'IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(),'hit',np.mean(r.ic>0),'turnover10_proxy',np.nanmean(turnover))
for name,m in [('all',np.ones(len(r),bool)),('vix_up',r.z>0),('vix_down',r.z<=0),('stress_z1',r.z>1)]:
 q=r.ic[m]; print(name,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std() if len(q)>1 else np.nan,'hit',np.mean(q>0) if len(q) else np.nan)
for h in [5,10,20]: print('decay',h,'not computed')
print('periods')
for y,g in r.groupby(r.index.year): print(y, len(g), round(g.ic.mean(),4),round(g.ic.mean()/g.ic.std(),4))
