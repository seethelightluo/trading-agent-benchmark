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
px=pd.concat(P,axis=1).sort_index(); r=px.pct_change()
mac=[]
for fnm in ['DXY.csv','VIX.csv','USDCNY.csv','USDJPY.csv','EURUSD.csv']:
    z=pd.read_csv('../persistent/index_data/'+fnm); z.date=pd.to_datetime(z.date)
    col='close' if 'close' in z else 'Close'; mac.append(z.set_index('date')[col].astype(float).rename(fnm[:-4]))
M=pd.concat(mac,axis=1).reindex(px.index).ffill().pct_change()
# Lagged rolling multivariate macro residual. Betas use data through t-1; signal at t uses residual through t.
res=pd.DataFrame(index=px.index,columns=px.columns,dtype=float)
for s in px:
    y=r[s]; z=pd.concat([y,M],axis=1).dropna()
    # rolling covariance regression, shifted so today's residual does not use today's macro beta
    bet=[]
    for c in M:
        bet.append(y.rolling(60,min_periods=40).cov(M[c])/M[c].rolling(60,min_periods=40).var())
    B=pd.concat(bet,axis=1); B.columns=M.columns
    pred=(B.shift(1)*M).sum(axis=1)
    res[s]=y-pred
sig=-res.rolling(5,min_periods=5).sum()
sig=sig.sub(sig.median(axis=1),axis=0)
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
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20310206_multimacro_residual_signal.csv',index=False)
