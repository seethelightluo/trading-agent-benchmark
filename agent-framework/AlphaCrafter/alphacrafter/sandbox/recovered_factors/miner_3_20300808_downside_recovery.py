import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
px={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date')['close'] for f in glob.glob('../persistent/stock_data/*.csv')}
p=pd.DataFrame(px).sort_index().astype(float)
ret=p.pct_change(); vol=ret.rolling(20,min_periods=12).std().shift(1)
# Downside-recovery persistence: only activate after a prior stressed drawdown;
# signal is lagged 5-observation rebound from the pre-stress 20d low, scaled by lagged volatility.
prior_low=p.rolling(20,min_periods=12).min().shift(6)
prior_high=p.rolling(60,min_periods=30).max().shift(1)
stress=p.shift(6)/prior_high-1 <= -0.08
rebound=p.shift(1)/prior_low-1
sig=(rebound/vol).where(stress)
# winsorize cross-sectionally to avoid crypto outliers
sig=sig.clip(-10,10)
print('candidate=downside_recovery_persistence_60_20_5obs')
print('dates',len(p),'instruments',len(p.columns),'coverage',round(sig.notna().sum().sum()/sig.size,6),'meanN',round(sig.notna().sum(axis=1).mean(),3))
for h in [1,5,10,20]:
 vals=[];ns=[]; f=p.shift(-h)/p-1
 for d in p.index:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.asarray(vals);print('horizon',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
print('turnover_proxy',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
for h in [5,10,20]:
 f=p.shift(-h)/p-1
 for label,mask in [('2020-23',p.index<'2024-01-01'),('2024-27',(p.index>='2024-01-01')&(p.index<'2028-01-01')),('2028+',p.index>='2028-01-01'),('latest120',p.index>=p.index[-120])]:
  a=[]
  for d in p.index[mask]:
   z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
   if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  a=np.asarray(a);print('regime',h,label,'dates',len(a),'IC',round(a.mean(),6) if len(a) else None,'ICIR',round(a.mean()/a.std(ddof=1),6) if len(a)>1 else None)
# simple decay and concentration
print('active_date_fraction',round(sig.notna().any(axis=1).mean(),6))
print('cross_section_median_active',round(sig.notna().sum(axis=1).median(),2))
