import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
b=Path('../persistent/stock_data')
P=pd.DataFrame({s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}).sort_index().loc[:'2028-09-20'].ffill(); R=P.pct_change(); bm=R.mean(axis=1)
beta=R.rolling(60,min_periods=30).cov(bm).div(bm.rolling(60,min_periods=30).var(),axis=0); vol=bm.rolling(20).std(); q75=vol.rolling(252,min_periods=126).quantile(.75)
f=-(P.pct_change(3)-beta.mul(bm.rolling(3).sum(),axis=0)); f[(bm.rolling(5).sum()<=0)|(vol>=q75)]=np.nan
rows=[]; ns=[]; valid=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],P.shift(-10).loc[dt]/P.loc[dt]-1],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic; rows.append(ic); ns.append(len(z)); valid.append((dt,ic))
a=np.array(rows); print('dates',len(a),'avgN',round(np.mean(ns),3),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'coverage',f.notna().mean().mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for lo,hi in [(2020,2022),(2023,2024),(2025,2026),(2027,2028)]:
 q=np.array([v for d,v in valid if lo<=d.year<=hi]); print(f'{lo}-{hi}','n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
for h in [1,5,10,20]:
 rr=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],P.shift(-h).loc[dt]/P.loc[dt]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(rr); print('decay',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
f.to_csv('scripts/miner_3_20280921_bull_q75_residual3_signal.csv')
