import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 d=get_stock_daily_data(s,2600)
 if d is None or len(d)<150:d=get_index_daily_data(s,2600)
 return d.set_index('date').close.rename(s)
# use union then ffill and restrict rows with genuine observations; daily async calendars are acceptable
close=pd.concat([get(s) for s in U],axis=1).sort_index().ffill()
ret=close.pct_change(); prior=close.shift(1)
rec=prior/prior.rolling(20,min_periods=15).min()-1
down=ret.where(ret<0).rolling(40,min_periods=25).std(); long=prior.pct_change(120)
f=rec/(down*np.sqrt(40)+1e-8)/(1+np.maximum(-long,0)); f=f.replace([np.inf,-np.inf],np.nan)
rows=[]
for i in range(len(close)-20):
 vals=f.iloc[i]
 if vals.notna().sum()<8: continue
 for h in [1,5,10,20]:
  y=close.iloc[i+h]/close.iloc[i]-1; z=pd.concat([vals.rename('x'),y.rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append((close.index[i],h,len(z),z.x.corr(z.y)))
r=pd.DataFrame(rows,columns=['date','h','n','ic'])
print('range',close.index.min(),close.index.max(),'dates',r.date.nunique(),'assets',len(close.columns))
for h in [1,5,10,20]:
 q=r[r.h==h].groupby('date').ic.first().dropna(); print(h,'dates',len(q),'avgN',r[r.h==h].n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
for yr in [2024,2025,2026,2027,2028,2029,2030]:
 q=r[(r.h==10)&(r.date.dt.year==yr)].groupby('date').ic.first();
 if len(q): print('yr',yr,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
sig=f.rank(axis=1,pct=True); print('coverage',f.notna().mean().mean(),'turnover',sig.diff().abs().mean().mean())
f.to_csv('scripts/miner_3_20301017_drawdown_recovery_signal.csv')
