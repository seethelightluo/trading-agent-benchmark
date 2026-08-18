import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-02-18'); base=Path('../persistent/stock_data')
px={s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
P=pd.DataFrame(px).sort_index().loc[:end].ffill(); r3=P.pct_change(3)
raw=-(r3.sub(r3.median(axis=1),axis=0))
# Activate short-term relative reversal only on unusually dispersed cross-sections.
disp=r3.sub(r3.median(axis=1),axis=0).abs().median(axis=1)
active=disp.gt(disp.rolling(60,min_periods=30).median())
f=raw.where(active, np.nan)
y=P.shift(-10)/P-1

def calc(x, lo=None, hi=None):
 q=x if lo is None else x.loc[lo:hi]; a=[]; ns=[]
 for dt in q.index:
  z=pd.concat([q.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.asarray(a); return len(a),float(np.mean(ns)),float(a.mean()),float(a.mean()/a.std(ddof=1)),float(np.mean(a>0))
print('candidate dispersion_conditioned_3d dates/N/IC/ICIR/hit',calc(f))
print('raw baseline',calc(raw))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-02-18')]:
 print('REG',lo,calc(f,lo,hi))
print('active_date_fraction',float(active.mean()),'coverage',float(f.notna().sum(axis=1).ge(8).mean()),'turnover',float(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()),'period',P.index.min().date(),P.index.max().date())
f.to_csv('scripts/miner_1_20280229_dispersion_conditioned_3d_signal.csv')
