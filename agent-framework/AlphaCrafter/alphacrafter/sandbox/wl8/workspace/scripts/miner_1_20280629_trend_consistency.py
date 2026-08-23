import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-06-28')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').sort_index()
idx=sorted(set.intersection(*[set(v.index) for v in P.values()])); cl=pd.DataFrame({s:P[s].reindex(idx).close for s in U}); r=cl.pct_change()
# Trend consistency: lagged 20-session return scaled by the fraction of up sessions; completed data only.
ret20=r.rolling(20).sum().shift(1); consistency=r.gt(0).rolling(20).mean().shift(1)
sig=ret20*(0.5+consistency)
fwd={h:cl.shift(-h)/cl-1 for h in [1,3,5,10]}
def eval(sig,f):
 vals=[];ds=[];ns=[]
 for d in idx:
  g=pd.DataFrame({'s':sig.loc[d],'f':f.loc[d]}).dropna()
  if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1:
   q=spearmanr(g.s,g.f).statistic
   if np.isfinite(q): vals.append(q);ds.append(d);ns.append(len(g))
 a=np.array(vals); return a,ds,ns
print('idea=trend_consistency_20d');
a,ds,ns=eval(sig,fwd[1]); print('dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(sig.notna().sum().sum()/sig.size,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
for h,f in fwd.items():
 z,_,_=eval(sig,f); print('horizon',h,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
for lab,fn in {'2020-22':lambda z:z.year<=2022,'2023-25':lambda z:2023<=z.year<=2025,'2026':lambda z:z.year==2026,'2027':lambda z:z.year==2027,'2028':lambda z:z.year>=2028,'recent180':lambda z:z>=END-pd.Timedelta(days=180)}.items():
 z=a[[i for i,x in enumerate(ds) if fn(x)]]; print(lab,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
