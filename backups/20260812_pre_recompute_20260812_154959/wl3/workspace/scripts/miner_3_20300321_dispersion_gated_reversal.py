import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for fn in (get_stock_daily_data,get_index_daily_data):
        try:
            x=fn(s,days=2600)
            if x is not None and len(x)>0:return x
        except Exception: pass
p={s:fetch(s) for s in U}; c=pd.concat({s:x.set_index('date')['close'] for s,x in p.items() if x is not None},axis=1).sort_index()
r=np.log(c).diff(); vol=r.rolling(30).std();
# Contrarian five-session move, volatility scaled, activated only on high cross-sectional dispersion.
move=np.log(c.shift(3)/c.shift(8)); base=-move/(vol*np.sqrt(5))
disp=r.rolling(5).std().median(axis=1) # common cross-asset turbulence proxy
disp_z=(disp-disp.rolling(120).median())/(disp.rolling(120).std())
raw=base.where(disp_z>0.0,0.0)
sig=raw.sub(raw.median(axis=1),axis=0).shift(1)
sig.to_csv('scripts/miner_3_20300321_dispersion_gated_reversal_signal.csv')
print('dates',len(c),'instruments',c.shape[1],'date_range',c.index.min(),c.index.max())
for h in [1,3,5,10]:
 f=np.log(c.shift(-h)/c); q=[]; ns=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 q=pd.Series(q).dropna();print('h',h,'obs',len(q),'avg_n',np.mean(ns),'ic',q.mean(),'icir',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
print('coverage',sig.notna().sum(axis=1).mean()/len(U),'rank_turnover',sig.rank(pct=True).diff().abs().mean().mean())
# regime halves
f=np.log(c.shift(-5)/c); qd=[]
for d in sig.index:
 z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
 if len(z)>=8:qd.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
q=pd.DataFrame(qd,columns=['date','ic']).set_index('date');
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2027-12-31'),('2028-01-01','2030-03-20')]:
 x=q.loc[a:b,'ic'];print('regime',a,b,'obs',len(x),'ic',x.mean(),'icir',x.mean()/x.std(ddof=1) if len(x)>1 else np.nan)
