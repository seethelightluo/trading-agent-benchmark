import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-02-28'); base=Path('../persistent/stock_data')
px={s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
P=pd.DataFrame(px).sort_index().loc[:end].ffill(); y=P.shift(-10)/P-1
r3,r5,r10=P.pct_change(3),P.pct_change(5),P.pct_change(10)
vol=P.pct_change().rolling(20).std()*np.sqrt(252)
raw=(0.5*r3+0.3*r5+0.2*r10)/vol
f=-raw.sub(raw.median(axis=1),axis=0)
def calc(start=None,stop=None):
 a=[];ns=[]
 for dt in f.index:
  if start and dt<pd.Timestamp(start):continue
  if stop and dt>pd.Timestamp(stop):continue
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.asarray(a);return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)
print('volscaled_smoothed ALL',calc())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-02-28')]:print('REG',lo,hi,calc(lo,hi))
rk=f.rank(axis=1,pct=True)
print('coverage',f.notna().sum(axis=1).ge(8).mean(),'turnover',((rk-rk.shift()).abs().mean(axis=1).dropna().mean()),'avg_valid_n',f.notna().sum(axis=1).mean())
f.to_csv('scripts/miner_3_20280229_volscaled_smoothed_signal.csv')
