import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-02-24'); base=Path('../persistent/stock_data')
px={s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
P=pd.DataFrame(px).sort_index().loc[:end].ffill(); R=P.pct_change()
# Candidate: multi-horizon contrarian blend, equal combination of 3d and 10d relative returns.
r3=P.pct_change(3); r10=P.pct_change(10)
f=-(0.5*r3.sub(r3.median(axis=1),axis=0)+0.5*r10.sub(r10.median(axis=1),axis=0))
y=P.shift(-10)/P-1
def calc(x, sl=slice(None)):
 a=[]; ns=[]
 for dt in x.loc[sl].index:
  z=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.asarray(a); return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)
for name,x in [('blend',f),('r3',-r3.sub(r3.median(axis=1),axis=0)),('r10',-r10.sub(r10.median(axis=1),axis=0))]:
 print(name,'ALL dates meanN IC ICIR hit',tuple(round(v,6) if isinstance(v,float) else v for v in calc(x)))
 for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-02-24')]:
  q=calc(x,slice(lo,hi)); print('REG',lo,hi,'dates',q[0],'N',round(q[1],2),'IC',round(q[2],6),'ICIR',round(q[3],6))
rk=f.rank(axis=1,pct=True); print('coverage',round(f.notna().sum(axis=1).ge(8).mean(),4),'turnover',round((rk-rk.shift()).abs().mean(axis=1).dropna().mean(),4),'period',P.index.min().date(),P.index.max().date())
f.to_csv('scripts/miner_3_20280225_multihorizon_reversal_signal.csv')
