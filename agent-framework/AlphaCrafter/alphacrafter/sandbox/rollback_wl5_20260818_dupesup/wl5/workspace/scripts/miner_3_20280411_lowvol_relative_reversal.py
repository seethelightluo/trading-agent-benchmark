import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-04-10'); base=Path('../persistent/stock_data')
px={s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
P=pd.DataFrame(px).sort_index().loc[:end].ffill(); R=P.pct_change(); r3=P.pct_change(3); vol=R.rolling(20).std()
# Low-volatility-conditioned relative reversal: contrarian 3d relative return, scaled by inverse trailing 20d volatility.
rel=r3.sub(r3.median(axis=1),axis=0)
f=-rel/(vol+1e-8)
y=P.shift(-10)/P-1
ics=[]; ns=[]; dates=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
a=np.asarray(ics)
print('dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-04-10')]:
 q=(np.array(dates)>=pd.Timestamp(lo))&(np.array(dates)<=pd.Timestamp(hi)); b=a[q]
 print('REG',lo,hi,'dates',len(b),'N',np.mean(np.array(ns)[q]) if len(b) else 0,'IC',b.mean() if len(b) else np.nan,'ICIR',b.mean()/b.std(ddof=1) if len(b)>1 else np.nan,'hit',np.mean(b>0) if len(b) else np.nan)
rk=f.rank(axis=1,pct=True); print('coverage',f.notna().sum(axis=1).ge(8).mean(),'turnover', (rk-rk.shift()).abs().mean(axis=1).dropna().mean(),'period',P.index.min().date(),P.index.max().date())
# decay horizons
for h in [1,3,5,10,20]:
 yy=P.shift(-h)/P-1; aa=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('DECAY',h,np.mean(aa),np.mean(aa)/np.std(aa,ddof=1))
f.to_csv('scripts/miner_3_20280411_lowvol_relative_reversal_signal.csv')
