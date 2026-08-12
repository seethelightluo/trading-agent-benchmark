import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<120:d=get_index_daily_data(s,days=3000)
 if d is not None:D[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').sort_index().close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
# Defensive downside asymmetry, with a cross-asset stress gate and one-day lag.
v=r.rolling(20,min_periods=20).std(); dn=r.clip(upper=0).pow(2).rolling(20,min_periods=20).mean().pow(.5)
stress=(r.mean(axis=1).rolling(20,min_periods=15).std()/r.mean(axis=1).rolling(120,min_periods=60).std())
thr=stress.rolling(252,min_periods=100).quantile(.65)
f=((1-dn/v)*stress.where(stress>thr,0)).shift(1)
for h in [1,3,5,10]:
 a=[];ns=[]; Y=np.log(p).shift(-h)-np.log(p)
 for dt in f.index:
  z=pd.concat([f.loc[dt],Y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q):a.append(q);ns.append(len(z))
 a=np.array(a);print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('coverage',round(f.notna().sum(axis=1).mean()/len(U),4),'rank_turnover',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.05).mean(),4))
