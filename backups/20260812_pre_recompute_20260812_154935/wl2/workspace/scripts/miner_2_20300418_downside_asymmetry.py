import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,2400)
 if x is None or len(x)<100: x=get_index_daily_data(s,2400)
 if x is not None and len(x): D[s]=x.set_index('date')['close'].astype(float)
px=pd.DataFrame(D).sort_index().ffill(); r=px.pct_change()
# downside-asymmetry reversal: recent selloff relative to downside risk, with medium-term rebound anchor
# positive signal means prefer assets with unusually bad recent return when downside risk is high
rv=r.rolling(20,min_periods=12).std(); down=np.sqrt((r.clip(upper=0)**2).rolling(20,min_periods=12).mean())
f=(-(r.rolling(3,min_periods=3).sum())/(down*np.sqrt(3))).add(0.35*r.rolling(20,min_periods=15).sum()).shift(1)
rows=[]
for i in range(len(px)-10):
 for h in [1,5,10]:
  z=pd.concat([f.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: rows.append((px.index[i],h,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
a=pd.DataFrame(rows,columns=['date','h','n','ic']).dropna()
for h in [1,5,10]:
 z=a[a.h==h].ic; print('h',h,'dates',len(z),'avgN',a[a.h==h].n.mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1)*np.sqrt(len(z)),'hit',(z>0).mean())
print('assets',len(D),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for name,mask in [('2020-22',a.date<'2023-01-01'),('2023-25',(a.date>='2023-01-01')&(a.date<'2026-01-01')),('2026-30',a.date>='2026-01-01')]:
 z=a[(a.h==1)&mask].ic;print(name,len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1)*np.sqrt(len(z)) if len(z)>1 else np.nan)
print('last',px.index[-1])
f.to_csv('scripts/miner_2_20300418_downside_asymmetry_signal.csv')
