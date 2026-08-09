import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
D={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in glob.glob('../persistent/stock_data/*.csv')}
p=pd.DataFrame({k:v.close for k,v in D.items()}).sort_index().astype(float); v=pd.DataFrame({k:x.volume for k,x in D.items()}).reindex(p.index).astype(float)
r=p.pct_change()
# One-day lagged volume-shock reversal: fade the prior day's move only when
# participation was unusually high, normalized by 20d volatility.
vr=(v.shift(1)/v.shift(1).rolling(60,min_periods=30).median()).clip(.25,6)
vol=r.shift(1).rolling(20,min_periods=10).std()
sig=(-r.shift(1)*np.log(vr)/vol).replace([np.inf,-np.inf],np.nan)
fwd={h:p.shift(-h)/p-1 for h in [1,5,10,20]}
print('candidate=volume_shock_reversal_1_60_volnorm');print('dates',len(p),'instruments',len(p.columns),'coverage',round(sig.notna().sum().sum()/sig.size,6),'meanN',round(sig.notna().sum(axis=1).mean(),2))
for h,f in fwd.items():
 a=[];ns=[]
 for d in p.index:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):a.append(q);ns.append(len(z))
 a=np.array(a);print('horizon',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
for h in [1,5,10,20]:
 print('regimes',h,end=' ')
 for lab,m in [('pre24',p.index<'2024-01-01'),('24_27',(p.index>='2024-01-01')&(p.index<'2028-01-01')),('latest',p.index>=p.index[-120])]:
  a=[]
  for d in p.index[m]:
   z=pd.concat([sig.loc[d],fwd[h].loc[d]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  a=np.array(a);print(lab+':'+str(len(a))+','+str(round(a.mean(),5) if len(a) else None),end=' ')
 print()
print('turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
for lag in [1,5,10]:
 z=pd.concat([sig.stack(),sig.shift(lag).stack()],axis=1).dropna();print('decay',lag,round(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,5),len(z))
