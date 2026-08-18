import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-03-13'); base=Path('../persistent/stock_data')
d={s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:x.close for s,x in d.items()}).sort_index().loc[:end].ffill()
O=pd.DataFrame({s:x.open for s,x in d.items()}).reindex(P.index).ffill()
gap=O/P.shift(1)-1
f=-gap.sub(gap.median(axis=1),axis=0); y=P.shift(-10)/P-1

def calc(x,lo=None,hi=None):
 a=[]; ns=[]
 for dt in x.index:
  if lo and dt<pd.Timestamp(lo): continue
  if hi and dt>pd.Timestamp(hi): continue
  z=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(ic): a.append(ic); ns.append(len(z))
 a=np.asarray(a)
 return len(a),round(float(np.mean(ns)),2),float(a.mean()),float(a.mean()/a.std(ddof=1)),float(np.mean(a>0))
print('overnight_gap_reversal ALL dates avgN IC ICIR hit',calc(f))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-03-13')]: print('REG',lo,hi,calc(f,lo,hi))
rk=f.rank(axis=1,pct=True)
print('coverage',float(f.notna().sum(axis=1).ge(8).mean()),'turnover',float((rk-rk.shift()).abs().mean(axis=1).dropna().mean()),'avg_valid_n',float(f.notna().sum(axis=1).mean()),'period',P.index.min().date(),P.index.max().date())
f.to_csv('scripts/miner_1_20280314_overnight_gap_reversal_signal.csv')
