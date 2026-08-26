import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
cut=pd.Timestamp('2034-11-27'); px={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.rsplit('/',1)[1][:-4];d=pd.read_csv(f);d.date=pd.to_datetime(d.date);px[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); market=r.mean(axis=1)
# residual medium-term reversal: asset's 20d return less equal-weight market return, risk scaled
res=p.pct_change(20).sub(p.pct_change(20).mean(axis=1),axis=0)
f=(-res/(r.rolling(20).std()*np.sqrt(252))).shift(1)
f.to_csv('scripts/miner_2_20341127_residual_reversal_signal.csv')
for h in [1,5,10,20]:
 rr=p.pct_change(h).shift(-h); a=[]; ns=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],rr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(ic):a.append(ic);ns.append(len(z))
 a=np.array(a);print(h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
print('coverage',f.notna().mean().mean(),'recent10',end=' ')
rr=p.pct_change(10).shift(-10);a=[]
for dt in p.index:
 z=pd.concat([f.loc[dt],rr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):a.append(q)
a=np.array(a);print(a[-500:].mean(),a[-500:].mean()/a[-500:].std(ddof=1))
