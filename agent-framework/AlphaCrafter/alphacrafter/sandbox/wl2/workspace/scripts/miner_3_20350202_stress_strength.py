import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D=['XAU','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A if os.path.exists('../persistent/stock_data/'+a+'.csv')}
P=pd.DataFrame(p).sort_index(); r=P.pct_change(); db=r[D].mean(axis=1)
beta=r.rolling(90,min_periods=60).cov(db).div(db.rolling(90,min_periods=60).var(),axis=0).shift(1)
res=r.sub(beta.mul(db,axis=0)); rv=res.rolling(60,min_periods=40).std().shift(1)
base=-res.rolling(30,min_periods=20).sum().shift(1)/(rv*np.sqrt(30)+1e-9)
# Smooth stress strength: require lagged defensive-basket return to exceed a volatility-scaled threshold.
dret=db.rolling(60,min_periods=40).sum().shift(1); dvol=db.rolling(60,min_periods=40).std().shift(1)*np.sqrt(60)
z=dret/(dvol+1e-12)
y=P.pct_change(40).shift(-40)
for th in [0.0,0.25,0.5,0.75,1.0]:
 sig=base.where(z>th); vals=[];ns=[];ds=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8: vals.append(spearmanr(sig.loc[dt][ok],y.loc[dt][ok]).statistic);ns.append(ok.sum());ds.append(dt)
 q=pd.Series(vals,index=ds)
 recent=q.loc['2031':'2035']; late=q.loc['2034':'2035']
 print('threshold',th,'dates',len(q),'avg_n %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()),'coverage %.4f'%(sig.notna().mean().mean()),'active %.4f'%(z>th).mean(),'recentIC %.6f'%recent.mean(),'lateIC %.6f'%late.mean())
 # Save only the best candidate after inspection is not done here; artifacts saved for candidates are not admission.
