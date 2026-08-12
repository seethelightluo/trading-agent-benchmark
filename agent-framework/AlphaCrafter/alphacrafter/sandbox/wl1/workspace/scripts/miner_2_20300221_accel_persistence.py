import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0: d=get_index_daily_data(s,5000)
 if d is None:return None
 d=d.copy(); d['date']=pd.to_datetime(d.date); d=d.set_index('date').sort_index()
 return d.close.astype(float)
px={s:load(s) for s in U}; px={s:x for s,x in px.items() if x is not None}
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# acceleration: recent 20d trend relative to its 60d baseline, scaled by 40d realized vol.
# Signal at date t uses close through t; evaluation forward returns begin t+1.
trend20=P/P.shift(20)-1; trend60=P/P.shift(60)-1
sig=(trend20-trend60/3)/(r.rolling(40).std()*np.sqrt(20))
# suppress unstable signals using sign agreement with 60d trend (interpretable persistence gate)
sig=sig.where(np.sign(trend20)==np.sign(trend60),0.0).shift(1)
print('rows',len(P),'assets',len(P.columns),'range',P.index.min(),P.index.max())
for h in [1,5,10,20]:
 fwd=P.shift(-h)/P-1
 z=[]
 for dt in sig.index:
  a=sig.loc[dt]; b=fwd.loc[dt]
  q=pd.concat([a,b],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 z=pd.Series(z).dropna()
 print(h,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1)*np.sqrt(len(z)),'hit',(z>0).mean())
# coverage, turnover, regime 2028+
valid=sig.notna().sum(axis=1); print('coverage', (valid/len(P.columns)).mean(),'avg_n',valid.mean(),'turnover',sig.diff().abs().mean().mean())
for label,lo,hi in [('2020-25','2020','2025-12-31'),('2026-28','2026','2028-12-31'),('2029+','2029','2100')]:
 fwd=P.shift(-10)/P-1; z=[]
 for dt in sig.loc[lo:hi].index:
  q=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 z=pd.Series(z).dropna(); print(label,len(z),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(len(z)) if len(z)>1 else np.nan)
sig.to_csv('scripts/miner_2_20300221_accel_persistence_signal.csv',index_label='date')
