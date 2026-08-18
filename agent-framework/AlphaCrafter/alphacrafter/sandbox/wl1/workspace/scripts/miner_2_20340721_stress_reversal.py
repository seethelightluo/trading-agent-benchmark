import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d):px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(px).sort_index().ffill()
v=get_index_daily_data('VIX',5000); V=v.set_index(pd.to_datetime(v.date)).close.astype(float).reindex(P.index).ffill() if v is not None else pd.Series(index=P.index,dtype=float)
# High-stress conditional short-horizon reversal, scaled by idiosyncratic recent vol; inactive calm dates set to cross-sectional neutral.
r=P.pct_change(5); vol=P.pct_change().rolling(20).std(); cross=P.pct_change().mean(axis=1).rolling(20).std()
# stress defined relative to trailing 120d VIX median, fully lagged
stress=(V>V.rolling(120).median()).astype(float).shift(1)
sig=((-r/vol.replace(0,np.nan))*stress.values[:,None]).shift(1)
fwd=P.shift(-10)/P-1
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].notna().sum()>=8: rows.append(z.iloc[:,0].corr(z.iloc[:,1]))
x=pd.Series(rows).dropna();print('dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(),6),'hit',round((x>0).mean(),4))
for h in [5,20,40]:
 y=[]; ff=P.shift(-h)/P-1
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8:y.append(z.iloc[:,0].corr(z.iloc[:,1]))
 y=pd.Series(y).dropna(); print('H',h,'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(),6))
print('coverage',round(sig.notna().sum(axis=1).mean()/len(U),4),'active',round(stress.mean(),4))
sig.to_csv('scripts/miner_2_20340721_stress_reversal_signal.csv',index_label='date')
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 y=[]
 for dt in sig.loc[a:b].index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:y.append(z.iloc[:,0].corr(z.iloc[:,1]))
 y=pd.Series(y).dropna();print(a,b,len(y),round(y.mean(),6),round(y.mean()/y.std(),6) if len(y)>1 else np.nan)
