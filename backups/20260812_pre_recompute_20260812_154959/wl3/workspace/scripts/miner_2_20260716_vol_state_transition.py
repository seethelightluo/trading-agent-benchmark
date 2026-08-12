import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
paths=glob.glob('../persistent/stock_data/*.csv')
D={}
for p in paths:
 s=p.split('/')[-1][:-4]; x=pd.read_csv(p); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close
px=pd.concat(D,axis=1).sort_index(); r=px.pct_change()
# Volatility compression/expansion: negative short/long realized-vol ratio
f=-(r.rolling(5,min_periods=5).std()/r.rolling(60,min_periods=40).std())
for h in [1,5,10]:
 fr=px.shift(-h)/px-1; vals=[]; dates=[]; ns=[]
 for dt in f.index:
  a=f.loc[dt]; b=fr.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
  if len(z)>=8:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(z))
 q=pd.Series(vals,index=dates).dropna()
 print('h',h,'dates',len(q),'avgN',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit', (q>0).mean())
 if h==1:
  # rank turnover daily
  ranks=f.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean(axis=1).mean(),'coverage',np.mean([f.loc[d].notna().mean() for d in f.index]))
# regime halves
for label,ix in [('early',q.index<'2023-01-01'),('late',q.index>='2023-01-01')]:
 z=q[ix]; print(label,len(z),z.mean(),z.mean()/z.std(ddof=1))
print('corr with 20d momentum', f.stack().corr(r.rolling(20).sum().stack()))
