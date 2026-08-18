import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-04-24'); base=Path('../persistent/stock_data')
d={s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:x.close for s,x in d.items()}).sort_index().loc[:end].ffill()
r=P.pct_change(); f=(P/P.shift(30)-1)/(r.rolling(20).std()*np.sqrt(20)); y=P.shift(-10)/P-1

def run(lo=None,hi=None):
 a=[];ns=[]
 for dt in P.index:
  if (lo and dt<pd.Timestamp(lo)) or (hi and dt>pd.Timestamp(hi)): continue
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): a.append(q); ns.append(len(z))
 a=np.asarray(a); return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)
print('ALL',run())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-04-24')]: print('REG',lo,run(lo,hi))
rk=f.rank(axis=1,pct=True); print('coverage',f.notna().sum(axis=1).ge(8).mean(),'turnover',((rk-rk.shift()).abs().mean(axis=1).dropna().mean()),'avgN',f.notna().sum(axis=1).mean())
f.to_csv('scripts/miner_1_20280425_risk_momentum30_signal.csv')
