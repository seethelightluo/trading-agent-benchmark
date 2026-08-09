import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}).sort_index()
R=P.pct_change()
# Volatility-managed momentum: 20d return divided by trailing 20d realized volatility, favoring strong risk-adjusted trends
F=P.pct_change(20)/(R.rolling(20).std()*np.sqrt(20))
for h in [1,5,10]:
 vals=[]; ns=[]; ds=[]
 for dt in F.index:
  y=P.shift(-h).loc[dt]/P.loc[dt]-1
  z=pd.concat([F.loc[dt],y],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); ds.append(dt)
 a=np.array(vals)
 print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026-09-10')]:
 a=[]
 for dt in F.index[(F.index>=lo)&(F.index<=hi)]:
  z=pd.concat([F.loc[dt],P.shift(-1).loc[dt]/P.loc[dt]-1],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a);print('REG',lo,hi,len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6))
print('coverage',round(F.notna().mean().mean(),4),'turnover',round(F.rank(pct=True,axis=1).diff().abs().mean(axis=1).mean(),4),'period',P.index.min().date(),P.index.max().date())
