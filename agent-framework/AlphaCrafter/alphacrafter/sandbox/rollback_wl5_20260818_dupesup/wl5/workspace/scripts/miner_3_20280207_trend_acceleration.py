import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-02-06'); b=Path('../persistent/stock_data')
px={s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
P=pd.DataFrame(px).sort_index().loc[:end].ffill(); y=P.shift(-10)/P-1
# Trend acceleration/exhaustion: recent 10d return relative to prior 10d return, demeaned cross-section.
r10=P.pct_change(10); prior10=P.shift(10).pct_change(10); f=(r10-prior10); f=f.sub(f.median(axis=1),axis=0)
def calc(x,lo=None,hi=None):
 a=[]; ns=[]
 for dt in x.loc[lo:hi].index:
  z=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.asarray(a); return len(a),float(np.mean(ns)),float(a.mean()),float(a.mean()/a.std(ddof=1)),float(np.mean(a>0))
print('candidate trend acceleration | end',end.date()); print('ALL',calc(f))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-02-06')]: print('REG',lo,hi,calc(f,lo,hi))
rk=f.rank(axis=1,pct=True); print('coverage',float(f.notna().sum(axis=1).ge(8).mean()),'turnover',float((rk-rk.shift()).abs().mean(axis=1).dropna().mean()))
f.to_csv('scripts/miner_3_20280207_trend_acceleration_signal.csv')
