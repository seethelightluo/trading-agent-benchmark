import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s,days=3000)
    if x is None or len(x)<100: x=get_index_daily_data(s,days=3000)
    if x is not None and len(x): D[s]=x.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Path-quality momentum: signed 20d return divided by total absolute daily movement,
# then volatility-normalize. High values identify persistent directional trends rather than noisy rallies.
ret20=p.pct_change(20); path=r.abs().rolling(20,min_periods=15).sum(); vol=r.rolling(40,min_periods=20).std()
f=ret20.div(path.replace(0,np.nan)).div(vol.replace(0,np.nan))
rows=[]
for i in range(len(p)-1):
    z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
    if len(z)>=8 and z.f.nunique()>1: rows.append((p.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); x=a.ic
print('dates',len(x),'avgN',a.n.mean(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for nm,mask in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-30',a.index>='2026-01-01')]:
 q=a.loc[mask].ic; print(nm,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [3,5,10]:
 y=p.pct_change(h).shift(-h+1); q=[]
 for i in range(len(p)-h):
    z=pd.concat([f.iloc[i],y.iloc[i]],axis=1).dropna()
    if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',h,'IC',np.nanmean(q),'ICIR',np.nanmean(q)/np.nanstd(q,ddof=1),'n',len(q))
f.to_csv('scripts/miner_2_20300808_path_quality_momentum_signal.csv')
