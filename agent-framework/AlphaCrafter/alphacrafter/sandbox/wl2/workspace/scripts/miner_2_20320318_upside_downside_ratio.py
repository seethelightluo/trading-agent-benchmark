import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
    d=get_stock_daily_data(s,4500)
    if d is None or len(d)<100: d=get_index_daily_data(s,4500)
    return d
xs={s:get(s) for s in U}
prices=pd.concat({s:d.set_index('date').close for s,d in xs.items() if d is not None},axis=1).sort_index().ffill()
ret=prices.pct_change()
# one idea: upside participation / downside participation, risk adjusted by vol; lag naturally via shift
up=ret.where(ret>0,0).rolling(20).sum()
dn=(-ret.where(ret<0,0)).rolling(20).sum()
# smooth ratio, with momentum sign, intended high = stronger asymmetric upside
fac=np.log((up+1e-5)/(dn+1e-5))
fac=fac.replace([np.inf,-np.inf],np.nan).shift(1)
# forward returns horizons, daily IC
rows=[]
for dt in fac.index:
    if dt not in prices.index: continue
    i=prices.index.get_loc(dt)
    if i+10>=len(prices): continue
    f=fac.loc[dt]; fw=prices.iloc[i+1:i+2].iloc[0]/prices.iloc[i].iloc[0]-1
    z=pd.concat([f,fw],axis=1).dropna()
    if len(z)>=8:
        rows.append([dt,len(z),z.iloc[:,0].corr(z.iloc[:,1])])
r=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('universe',len(prices.columns),'dates',len(prices),'IC dates',len(r),'avg n',r.n.mean())
print('daily IC %.6f ICIR %.6f hit %.4f coverage %.4f turnover %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(),(r.ic>0).mean(),fac.notna().sum().sum()/(fac.shape[0]*len(U)), (fac.diff().abs()>0.15).sum().sum()/fac.notna().sum().sum()))
for a,b in [('2020','2022'),('2023','2025'),('2026','2031'),('2032','2032')]:
 q=r.loc[a:b]
 if len(q): print(a,b,len(q),'IC %.6f ICIR %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std()))
for h in [1,3,5,10]:
 rr=[]
 for dt in fac.index:
  i=prices.index.get_loc(dt)
  if i+h>=len(prices): continue
  z=pd.concat([fac.loc[dt],prices.iloc[i+h]/prices.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,np.nanmean(rr),len(rr))
# signal artifact
out=fac.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20320318_upside_downside_ratio_signal.csv',index=False)
