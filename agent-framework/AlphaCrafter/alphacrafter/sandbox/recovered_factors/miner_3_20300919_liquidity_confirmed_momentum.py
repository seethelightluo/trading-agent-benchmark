import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
D={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in glob.glob('../persistent/stock_data/*.csv')}
p=pd.DataFrame({k:v.close for k,v in D.items()}).sort_index().astype(float)
v=pd.DataFrame({k:v.volume for k,v in D.items()}).reindex(p.index).astype(float)
r=p.pct_change()
# Liquidity-confirmed medium momentum: lagged 20d return, scaled by abnormal turnover.
# Volume ratio is clipped to avoid isolated synthetic volume spikes; all shifted one day.
ret20=p.shift(1)/p.shift(21)-1
vr=(v.shift(1).rolling(5,min_periods=3).mean()/v.shift(1).rolling(60,min_periods=30).mean()).clip(0.25,4.0)
sig=ret20*np.log(vr)
# fallback is zero when volume unavailable, preserving coverage but no artificial price lookahead
sig=sig.where(vr.notna(),ret20*0.0)
print('candidate=liquidity_confirmed_medium_momentum_20_5_60')
print('dates',len(p),'instruments',len(p.columns),'coverage',round(sig.notna().sum().sum()/sig.size,6),'meanN',round(sig.notna().sum(axis=1).mean(),2))
for h in [1,5,10,20]:
 a=[]; ns=[]; f=p.shift(-h)/p-1
 for d in p.index:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.asarray(a);print('horizon',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
print('turnover_proxy',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
for h in [10,20]:
 f=p.shift(-h)/p-1
 for label,mask in [('2020-23',p.index<'2024-01-01'),('2024-27',(p.index>='2024-01-01')&(p.index<'2028-01-01')),('2028+',p.index>='2028-01-01'),('latest120',p.index>=p.index[-120])]:
  a=[]
  for d in p.index[mask]:
   z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
   if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  a=np.asarray(a); print('regime',h,label,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6) if len(a)>1 else None)
# signal persistence and components
for lag in [1,5,10,20]:
 z=pd.concat([sig.stack().rename('a'),sig.shift(lag).stack().rename('b')],axis=1).dropna();print('decay',lag,round(spearmanr(z.a,z.b).statistic,6),len(z))
