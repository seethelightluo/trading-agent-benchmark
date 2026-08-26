import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}
cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill()
ret=cl.pct_change()
# one-day lagged, medium-term contrarian signal, scaled by range position:
# fade 60d return; emphasize assets near lower end of their 120d rolling range
r60=cl.pct_change(60)
vol=ret.rolling(60).std()*np.sqrt(252)
lo=cl.rolling(120).min(); hi=cl.rolling(120).max()
pos=(cl-lo)/(hi-lo).replace(0,np.nan)
sig=((-r60/(vol+0.05))*(1.0+0.75*(0.5-pos))).shift(1)
# winsorize cross-section per date
sig=sig.clip(-5,5)
rows=[]
for h in [10,20,40,60]:
    fwd=cl.shift(-h)/cl-1
    vals=[]; dates=[]; ns=[]
    for dt in sig.index:
        a=sig.loc[dt]; b=fwd.loc[dt]; ok=a.notna()&b.notna()
        if ok.sum()>=8:
            vals.append(a[ok].corr(b[ok],method='spearman')); dates.append(dt); ns.append(ok.sum())
    x=pd.Series(vals,index=pd.DatetimeIndex(dates)).dropna()
    rows.append((h,len(x),x.mean(),x.mean()/x.std(ddof=1), (x>0).mean(),np.mean(ns)))
print('period',cl.index.min().date(),cl.index.max().date(),'assets',len(cl.columns))
for z in rows: print('H',z[0],'dates',z[1],'IC %.6f ICIR %.6f hit %.4f avgN %.2f'%z[2:])
# coverage and rank turnover
cov=sig.notna().sum(axis=1)/len(U)
rank=sig.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna().mean()
print('coverage %.6f turnover %.6f'% (cov.mean(),turn))
for name,mask in [('2027',sig.index.year==2027),('2028-29',sig.index.year.isin([2028,2029])),('2030',sig.index.year==2030),('2031-32',sig.index.year.isin([2031,2032])),('2033YTD',sig.index.year==2033)]:
    fwd=cl.shift(-60)/cl-1; xs=[]
    for dt in sig.index[mask]:
        ok=sig.loc[dt].notna()&fwd.loc[dt].notna()
        if ok.sum()>=8: xs.append(sig.loc[dt,ok].corr(fwd.loc[dt,ok],method='spearman'))
    xs=pd.Series(xs).dropna(); print(name,'dates',len(xs),'IC %.6f ICIR %.6f hit %.4f'%(xs.mean(),xs.mean()/xs.std(ddof=1), (xs>0).mean()) if len(xs)>1 else 'insufficient')
# save signal artifact for reproducibility
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20331013_range_position_reversal_signal.csv',index=False)
