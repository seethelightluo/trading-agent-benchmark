import pandas as pd,numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-10-22')
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index() for a in assets}
# Cross-sectional residual momentum: 20-observation return minus contemporaneous equal-weight benchmark return.
rets={a:D[a].close[D[a].index<=end].pct_change() for a in assets}
R=pd.concat(rets,axis=1).sort_index(); benchmark=R.median(axis=1,skipna=True)
fac=pd.DataFrame(index=R.index,columns=assets,dtype=float)
for a in assets:
    fac[a]=R[a].rolling(20,min_periods=15).sum()-benchmark.rolling(20,min_periods=15).sum()
close=pd.concat({a:D[a].close[D[a].index<=end] for a in assets},axis=1).sort_index()
for h in [1,5,10]:
 fwd=close.pct_change(h).shift(-h);ics=[];dates=[];ns=[]
 for dt in fac.index:
  x=fac.loc[dt].dropna();y=fwd.loc[dt].reindex(x.index).dropna();x=x.reindex(y.index)
  if len(x)>=8 and x.nunique()>1 and y.nunique()>1:
   q=spearmanr(x,y).statistic
   if np.isfinite(q):ics.append(q);dates.append(dt);ns.append(len(x))
 s=pd.Series(ics,index=dates);print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),5),'ICIR',round(s.mean()/s.std(),5),'hit',round((s>0).mean(),4))
 if h==1:
  for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
   q=s[(s.index.year>=lo)&(s.index.year<=hi)];print('REG',lo,hi,len(q),round(q.mean(),5),round(q.mean()/q.std(),5))
print('coverage',round(fac.notna().sum(axis=1).mean()/15,4),'turnover',round(fac.rank(pct=True).diff().abs().mean(axis=1).mean(),4))
out=fac.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20261022_idio_signal.csv',index=False)
