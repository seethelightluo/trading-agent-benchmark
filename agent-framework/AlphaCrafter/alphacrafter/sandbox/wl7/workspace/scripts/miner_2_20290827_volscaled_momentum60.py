import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); p[s]=d.set_index('date').close
p=pd.DataFrame(p).sort_index().ffill()
# Risk-adjusted intermediate momentum: 60d return divided by 20d realized volatility.
# All signals are shifted one completed day before forward returns.
r=p.pct_change(); vol=r.rolling(20).std()*np.sqrt(252); f=(p.pct_change(60)/vol).shift(1)
fr={h:p.pct_change(h).shift(-h) for h in [1,5,10,20]}
def calc(a,b,dates=None):
 vals=[]; ns=[]; turns=[]
 ix=f.index if dates is None else f.loc[dates].index
 for dt in ix:
  z=pd.concat([a.loc[dt],b.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 x=np.asarray(vals); return len(x),np.mean(ns),x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0)
for h in [1,5,10,20]:
 print('H',h,'all',calc(f,fr[h]))
for name,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_25','2023-01-01','2025-12-31'),('2026_28','2026-01-01','2028-12-31'),('2029_YTD','2029-01-01','2029-08-27')]:
 print(name,calc(f,fr[10],(f.index>=lo)&(f.index<=hi)))
# signal coverage and average daily rank turnover
valid=f.notna().sum(axis=1); ranks=f.rank(axis=1,pct=True); turnover=ranks.diff().abs().mean(axis=1).dropna()
print('coverage',valid.mean()/len(U),'avgN',valid.mean(),'turnover',turnover.mean(),'dates',len(f))
f.to_csv('scripts/miner_2_20290827_volscaled_momentum60_signal.csv',index_label='date')
