import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); files=glob.glob('../persistent/stock_data/*.csv'); C={}
for p in files:
 a=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); C[a]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); r=close.pct_change(); u=r['US10Y']
# rolling covariance computed explicitly to avoid pandas MultiIndex alignment
mu=r.rolling(40,min_periods=25).mean(); muv=u.rolling(40,min_periods=25).mean(); vu=((u-muv)**2).rolling(40,min_periods=25).mean()
beta=pd.DataFrame({a:((r[a]-mu[a])*(u-muv)).rolling(40,min_periods=25).mean()/vu for a in r.columns})
dbeta=beta-beta.shift(10); shock=u.rolling(20,min_periods=15).std().replace(0,np.nan); trigger=(u.abs()/shock).clip(0,3)
fac=(-dbeta).mul(trigger,axis=0); fac=fac.sub(fac.mean(axis=1),axis=0).div(fac.std(axis=1).replace(0,np.nan),axis=0); fac.to_csv('scripts/miner_1_20270325_beta_instability_signal.csv')
def ev(h):
 y=close.pct_change(h).shift(-h); vals=[];ns=[];ds=[]
 for dt in fac.index:
  x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1: vals.append(spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic);ns.append(len(x));ds.append(dt)
 s=pd.Series(vals,index=ds);return s,ns
print('assets',len(C),'rows',len(fac),'period',fac.index.min().date(),fac.index.max().date())
for h in [1,5,10]:
 s,n=ev(h); print('H',h,'dates',len(s),'avgN',round(np.mean(n),2),'IC %.7f ICIR %.7f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,hi,len(q),'IC %.7f ICIR %.7f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/len(C),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
