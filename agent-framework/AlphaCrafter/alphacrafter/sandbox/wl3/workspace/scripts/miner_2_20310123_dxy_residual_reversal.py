import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
    d=None
    for fn in (get_stock_daily_data,get_index_daily_data):
        try: d=fn(s,5000)
        except Exception: pass
        if d is not None and len(d): break
    if d is not None and len(d):
        z=d.copy(); z.date=pd.to_datetime(z.date); P[s]=z.set_index('date').close.astype(float).rename(s)
px=pd.concat(P,axis=1).sort_index()
macro=pd.read_csv('../persistent/index_data/DXY.csv')
macro['date']=pd.to_datetime(macro['date'])
col='close' if 'close' in macro else 'Close'
dxy=macro.set_index('date')[col].astype(float).reindex(px.index).ffill()
# Macro-residual reversal: assets that underperformed their DXY beta-adjusted move
# over 5 days are expected to mean-revert over the next day.
r=px.pct_change(); m=dxy.pct_change()
res=pd.DataFrame(index=px.index,columns=px.columns,dtype=float)
for s in px:
    z=pd.concat([r[s],m],axis=1).dropna()
    cov=z.iloc[:,0].rolling(60,min_periods=30).cov(z.iloc[:,1]); var=m.rolling(60,min_periods=30).var()
    beta=cov/var.replace(0,np.nan)
    res[s]=r[s]-beta*m
sig=-res.rolling(5,min_periods=5).sum()
sig=sig.sub(sig.median(axis=1),axis=0).replace([np.inf,-np.inf],np.nan)
rows=[]
for dt in sig.index:
    z=pd.concat([sig.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
    if len(z)>=8:
        c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
        if pd.notna(c): rows.append((dt,len(z),c))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); q=a.ic
print('dates',len(a),'median_n',a.n.median(),'coverage',a.n.sum()/(len(a)*15))
print('daily IC',q.mean(),'std',q.std(),'ICIR',q.mean()/q.std(),'hit',(q>0).mean())
for h in [3,5,10,20]:
 y=px.pct_change(h).shift(-h); vv=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): vv.append(c)
 ss=pd.Series(vv); print('h',h,'IC',ss.mean(),'ICIR',ss.mean()/ss.std(),'n',len(ss))
for nm,sl in [('2020-22',slice('2020','2022')),('2023-25',slice('2023','2025')),('2026-27',slice('2026','2027')),('2028-30',slice('2028','2030'))]:
 ss=a.loc[sl,'ic'].dropna(); print(nm,len(ss),ss.mean(),ss.mean()/ss.std())
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20310123_dxy_residual_reversal_signal.csv',index=False)
