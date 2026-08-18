import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-02-14'); base=Path('../persistent/stock_data')
px={s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
P=pd.DataFrame(px).sort_index().loc[:end].ffill(); R3=P.pct_change(3); R10=P.pct_change(10)
# Equal-weight cross-sectional relative contrarian blend; positive values imply expected stronger forward return.
f=-(0.5*R3.sub(R3.median(axis=1),axis=0)+0.5*R10.sub(R10.median(axis=1),axis=0))
y=P.shift(-10)/P-1

def calc(x, lo=None, hi=None):
 a=[]; ns=[]
 q=x.loc[lo:hi] if lo or hi else x
 for dt in q.index:
  z=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(v): a.append(v); ns.append(len(z))
 a=np.asarray(a)
 return len(a),float(np.mean(ns)),float(a.mean()),float(a.mean()/a.std(ddof=1)),float(np.mean(a>0))
print('UNIVERSE',len(U),'DATES',P.index.min().date(),P.index.max().date())
for name,x in [('blend',f),('r3',-R3.sub(R3.median(axis=1),axis=0)),('r10',-R10.sub(R10.median(axis=1),axis=0))]:
 print(name,'ALL dates meanN IC ICIR hit',calc(x))
 for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-02-14')]: print('REG',lo,hi,calc(x,lo,hi))
rk=f.rank(axis=1,pct=True); print('coverage',float(f.notna().sum(axis=1).ge(8).mean()),'turnover',float((rk-rk.shift()).abs().mean(axis=1).dropna().mean()))
f.to_csv('scripts/miner_1_20280215_multihorizon_reversal_signal.csv')
