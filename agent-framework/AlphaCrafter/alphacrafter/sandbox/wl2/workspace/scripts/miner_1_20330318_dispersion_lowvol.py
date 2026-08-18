import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,5000)
   if x is not None and len(x)>100:return x
  except:pass
D={s:ld(s) for s in U};D={s:x for s,x in D.items() if x is not None}
C=pd.DataFrame({s:x.set_index(pd.to_datetime(x.date)).close.astype(float) for s,x in D.items()}).sort_index().groupby(level=0).last().ffill();R=C.pct_change()
vol=R.rolling(20,min_periods=15).std().shift(1); disp=R.std(axis=1).rolling(20,min_periods=15).mean().shift(1); q=disp.rolling(252,min_periods=100).quantile(.7).shift(1)
# defensive low-vol preference only in broad high-dispersion regimes
f=(-np.log(vol)).mul((disp>q).astype(float),axis=0)
print('assets',len(D),'dates',len(C),'active',int((disp>q).sum()),'coverage',round(f.notna().sum(axis=1).replace(0,np.nan).mean()/len(D),4))
for h in [1,3,5,10]:
 fr=R.rolling(h).sum().shift(-h);a=[];ds=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()));ds.append(d)
 a=pd.Series(a,index=ds).dropna();print('H',h,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),6),'hit',round((a>0).mean(),4))
 if h==1 and len(a):print('early_late',round(a.iloc[:len(a)//2].mean(),6),round(a.iloc[len(a)//2:].mean(),6))
f.to_csv('scripts/miner_1_20330318_dispersion_lowvol_signal.csv',index_label='date')
