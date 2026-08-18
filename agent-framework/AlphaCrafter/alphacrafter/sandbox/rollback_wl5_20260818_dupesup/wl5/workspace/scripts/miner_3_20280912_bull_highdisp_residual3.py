import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
b=Path('../persistent/stock_data')
P=pd.DataFrame({s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}).sort_index().loc[:'2028-09-11'].ffill()
R=P.pct_change(); bm=R.mean(axis=1)
beta=R.rolling(60,min_periods=30).cov(bm).div(bm.rolling(60,min_periods=30).var(),axis=0)
y=P.shift(-10)/P-1
res=P.pct_change(3)-beta.mul(bm.rolling(3).sum(),axis=0)
f=-res
# Positive benchmark trend plus high cross-sectional dispersion; all regime inputs available at t.
disp=R.rolling(20).std().mean(axis=1)
high=disp > disp.rolling(252,min_periods=126).median()
bull=bm.rolling(5).sum()>0
f[~(bull & high)]=np.nan
rows=[]; ns=[]; valid=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  rows.append(ic); ns.append(len(z)); valid.append((dt,ic))
a=np.asarray(rows)
print('dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'coverage',f.notna().mean().mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for lo,hi in [(2020,2022),(2023,2024),(2025,2026),(2027,2028)]:
 q=np.array([v for d,v in valid if lo<=d.year<=hi]); print(f'{lo}-{hi}','n',len(q),'IC',q.mean() if len(q) else np.nan,'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
f.to_csv('scripts/miner_3_20280912_bull_highdisp_residual3_signal.csv')
print('artifact written')
