import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}
cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill()
ret=cl.pct_change(); r20=cl.pct_change(20); r60=cl.pct_change(60)
v20=ret.rolling(20).std()*np.sqrt(252); v60=ret.rolling(60).std()*np.sqrt(252)
lo180=cl.rolling(180).min(); hi180=cl.rolling(180).max(); pos180=((cl-lo180)/(hi180-lo180).replace(0,np.nan)).clip(0,1)
base=(-.55*r20/(v20+.05)-.45*r60/(v60+.05))*(1.20-pos180).clip(.30,1.20)
# Existing long-horizon range-asymmetry component; remove its daily cross-sectional linear component.
lo252=cl.rolling(252).min(); hi252=cl.rolling(252).max(); pos252=((cl-lo252)/(hi252-lo252).replace(0,np.nan)).clip(0,1)
old=(-r60/(v60+.05))*(1.25-1.5*pos252).clip(.25,1.25)
def resid(row,z):
    ok=row.notna()&z.notna()
    out=pd.Series(np.nan,index=row.index)
    if ok.sum()>=8:
        x=z[ok].values; y=row[ok].values
        den=np.sum((x-x.mean())**2)
        beta=np.sum((x-x.mean())*(y-y.mean()))/den if den>1e-12 else 0
        out.loc[ok]=y-(y.mean()+beta*(x-x.mean()))
    return out
sig=pd.DataFrame({dt:resid(base.loc[dt],old.loc[dt]) for dt in cl.index}).T.clip(-5,5).shift(1)
print('period',cl.index.min().date(),cl.index.max().date(),'assets',len(cl.columns))
for h in [10,20,40,60]:
 fwd=cl.shift(-h)/cl-1; xs=[]; ns=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&fwd.loc[dt].notna()
  if ok.sum()>=8:
   q=sig.loc[dt,ok].corr(fwd.loc[dt,ok],method='spearman')
   if pd.notna(q): xs.append(q); ns.append(ok.sum())
 x=pd.Series(xs); print('H',h,'dates',len(x),'IC %.6f ICIR %.6f hit %.4f avgN %.2f'%(x.mean(),x.mean()/x.std(ddof=1), (x>0).mean(),np.mean(ns)))
fwd=cl.shift(-60)/cl-1
for name,mask in [('2027',sig.index.year==2027),('2028-29',sig.index.year.isin([2028,2029])),('2030',sig.index.year==2030),('2031-32',sig.index.year.isin([2031,2032])),('2033YTD',sig.index.year==2033)]:
 xs=[]
 for dt in sig.index[mask]:
  ok=sig.loc[dt].notna()&fwd.loc[dt].notna()
  if ok.sum()>=8:
   q=sig.loc[dt,ok].corr(fwd.loc[dt,ok],method='spearman')
   if pd.notna(q): xs.append(q)
 x=pd.Series(xs); print(name,'dates',len(x),'IC %.6f ICIR %.6f hit %.4f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()) if len(x)>1 else 'insufficient')
print('coverage %.6f turnover %.6f'%(sig.notna().sum(axis=1).div(len(U)).mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20331208_residual_multihorizon_reversal_signal.csv',index=False)
