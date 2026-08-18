import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-01-12'); base=Path('../persistent/stock_data')
px={s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
P=pd.DataFrame(px).sort_index().loc[:end].ffill(); R=P.pct_change(); r5=P.pct_change(5); vol=R.rolling(20,min_periods=15).std()
# Volatility-normalized relative reversal: contrarian 5d cross-sectional return divided by each asset's trailing risk.
raw=-(r5.sub(r5.median(axis=1),axis=0)); f=raw/vol
Y=P.shift(-10)/P-1
ics=[]; ns=[]; dates=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],Y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
a=np.asarray(ics)
print('dates',len(a),'meanN',round(np.mean(ns),3),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),6))
print('coverage',round(f.notna().sum(axis=1).ge(8).mean(),6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6),'period',P.index.min().date(),end.date())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-01-02')]:
 q=[ics[i] for i,d in enumerate(dates) if pd.Timestamp(lo)<=d<=pd.Timestamp(hi)]
 print('REG',lo,hi,'dates',len(q),'IC',round(np.mean(q),6) if q else None,'ICIR',round(np.mean(q)/np.std(q,ddof=1),6) if len(q)>1 else None)
f.to_csv('scripts/miner_3_20280113_volnorm_relative_reversal_signal.csv')
