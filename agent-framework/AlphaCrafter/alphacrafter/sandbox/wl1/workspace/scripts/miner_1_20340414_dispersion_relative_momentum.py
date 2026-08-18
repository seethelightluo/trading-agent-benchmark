import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
U=get_account_dict()['watch_list']
# Dispersion-gated residual momentum: relative 20d return vs cross-sectional median,
# volatility scaled; use only information through t-1 and evaluate t to t+9.
P={}
for s in U:
    d=get_stock_daily_data(s, days=5200)
    if d is None or len(d)<120: d=get_index_daily_data(s, days=5200)
    P[s]=d.set_index('date')['close'].astype(float) if d is not None else pd.Series(dtype=float)
px=pd.DataFrame(P).sort_index().ffill()
ret=px.pct_change()
# signal at date t uses prices through t, then explicitly shifted one row for forward return
r20=px/px.shift(20)-1
vol=ret.rolling(20).std()*np.sqrt(252)
csmed=r20.median(axis=1)
disp=r20.sub(r20.median(axis=1),axis=0).abs().median(axis=1)
# dispersion gate is continuous: emphasize relative momentum in high dispersion, neutralize in low dispersion
sig=r20.sub(csmed,axis=0).div(vol.replace(0,np.nan)).mul((disp/disp.rolling(60).median()),axis=0)
sig=sig.shift(1)
fwd=px.shift(-10)/px-1
rows=[]; ics=[]
for dt in sig.index:
    a=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
    if len(a)>=8:
        ic=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
        if np.isfinite(ic): ics.append((dt,ic,len(a)))
        rows.append((dt,len(a)))
q=pd.DataFrame(ics,columns=['date','ic','n']).set_index('date')
print('dates',len(q),'avgN',q.n.mean(),'coverage',q.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1), (q.ic>0).mean(), sig.rank(axis=1,pct=True).diff().abs().stack().mean()))
for a,b in [('2020','2024'),('2025','2029'),('2030','2032'),('2033','2034')]:
 z=q.loc[a:b]; print(a,b,len(z), z.ic.mean() if len(z) else np.nan, z.ic.mean()/z.ic.std(ddof=1) if len(z)>1 else np.nan)
for h in [5,10,20,40]:
 fw=px.shift(-h)/px-1; z=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(a)>=8: z.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(z),len(z))
# artifact for deterministic audit
out=sig.copy(); out.index=out.index.strftime('%Y-%m-%d'); out.to_csv('scripts/miner_1_20340414_dispersion_relative_momentum_signal.csv')
