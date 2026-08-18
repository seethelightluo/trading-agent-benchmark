import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-01-10'); base=Path('../persistent/stock_data')
d={s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}; P=pd.DataFrame({s:x.close for s,x in d.items()}).sort_index().loc[:end].ffill(); r=P.pct_change(); vol=r.rolling(20).std(); f=P.pct_change(60)/vol; y=P.shift(-10)/P-1

def calc(lo=None,hi=None):
 a=[];ns=[]
 for dt in P.index:
  if lo and dt<pd.Timestamp(lo):continue
  if hi and dt>pd.Timestamp(hi):continue
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):a.append(q);ns.append(len(z))
 a=np.array(a); return len(a),round(np.mean(ns),2),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round(np.mean(a>0),4)
print('60d risk momentum',calc())
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-01-10')]:print(a,b,calc(a,b))
rk=f.rank(axis=1,pct=True);print('coverage',round(float(f.notna().sum(axis=1).ge(8).mean()),4),'turnover',round(float((rk-rk.shift()).abs().mean(axis=1).dropna().mean()),4),'avgN',round(float(f.notna().sum(axis=1).mean()),2))
f.to_csv('scripts/miner_1_20280111_risk_momentum60_signal.csv')
