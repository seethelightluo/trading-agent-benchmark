import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s,days=4000)
    if d is None or len(d)<150: d=get_index_daily_data(s,days=4000)
    if d is not None and len(d)>100:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.drop_duplicates('date').set_index('date').sort_index(); frames[s]=d
close=pd.DataFrame({s:d['close'] for s,d in frames.items()}); ret=np.log(close).diff()
rollret=ret.rolling(60).sum(); vol=ret.rolling(40).std()*np.sqrt(252); path=ret.abs().rolling(60).sum(); eff=rollret.abs()/path.replace(0,np.nan)
high=close.rolling(80).max(); dd=(close/high-1).clip(-1,0)
factor=(rollret/(vol+1e-8))*((1+dd).clip(.25,1))*(0.5+0.5*eff)
f=factor.shift(1); y=ret.shift(-1); ics=[]; ns=[]; turns=[]; prev=None; dates=[]
for dt in f.index:
 x=f.loc[dt]; z=y.loc[dt]; ok=x.notna()&z.notna()
 if ok.sum()>=8:
  ic=spearmanr(x[ok],z[ok]).statistic
  if np.isfinite(ic):
   ics.append(ic); ns.append(int(ok.sum())); dates.append(dt)
   r=x[ok].rank(pct=True)
   if prev is not None: turns.append(np.mean(np.abs(r[r.index.intersection(prev.index)]-prev[r.index.intersection(prev.index)])))
   prev=r
ics=np.array(ics); dates=pd.DatetimeIndex(dates)
print('symbols',len(frames),'dates',len(ics),'avgN',np.mean(ns),'coverage',np.mean(ns)/len(frames))
print('daily IC %.6f ICIR %.6f hit %.4f turnover %.4f'%(ics.mean(),ics.mean()/ics.std(ddof=1),np.mean(ics>0),np.mean(turns)))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2028-08-09')]:
 q=ics[(dates>=a)&(dates<=b)]; print(a,b,'n',len(q),'IC %.6f ICIR %.6f hit %.3f'%(q.mean(),q.mean()/q.std(ddof=1),np.mean(q>0)))
for h in [5,10,20]:
 yy=np.log(close).diff(h).shift(-h); ii=[]
 for dt in f.index:
  x=f.loc[dt]; z=yy.loc[dt]; ok=x.notna()&z.notna()
  if ok.sum()>=8: ii.append(spearmanr(x[ok],z[ok]).statistic)
 print('h',h,'n',len(ii),'IC %.6f ICIR %.6f'%(np.nanmean(ii),np.nanmean(ii)/np.nanstd(ii,ddof=1)))
f.index.name='date'; f.to_csv('scripts/miner_2_20280810_drawdown_recovery_signal.csv')
