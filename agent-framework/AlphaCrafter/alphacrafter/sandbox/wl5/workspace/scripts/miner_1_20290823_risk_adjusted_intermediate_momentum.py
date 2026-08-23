import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr

ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2029-08-23')
px={}
for a in ASSETS:
 d=pd.read_csv(Path('../persistent/stock_data')/(a+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index()
 px[a]=d[d.index<=END]
prices=pd.DataFrame(px).ffill()
ret=prices.pct_change()
# interpretable risk-adjusted intermediate momentum: 20d return divided by 20d realized vol,
# then remove cross-sectional mean each day to emphasize relative signal
mom=prices.pct_change(20)
vol=ret.rolling(20,min_periods=15).std()*np.sqrt(252)
factor=(mom/vol).replace([np.inf,-np.inf],np.nan)
factor=factor.sub(factor.mean(axis=1),axis=0)
rows=[]
for h in [5,10,20]:
 fwd=prices.shift(-h)/prices-1
 ics=[]; cov=[]; turnovers=[]
 for dt in factor.index:
  x=factor.loc[dt]; y=fwd.loc[dt]
  ok=x.notna()&y.notna()
  if ok.sum()>=8:
   ics.append(spearmanr(x[ok],y[ok]).statistic); cov.append(ok.mean())
 # turnover of daily ranks, measured mean abs rank change / universe
 ranks=factor.rank(axis=1,pct=True)
 turnovers=ranks.diff().abs().mean(axis=1).dropna().mean()
 arr=np.array(ics); ic=arr.mean(); ir=ic/(arr.std(ddof=1)+1e-12)*np.sqrt(len(arr))
 rows.append((h,len(arr),ic,ir,np.mean(np.array(ics)>0),np.mean(cov),turnovers))
 # regimes
 for lo,hi in [('2020-01-01','2023-12-31'),('2024-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2029-01-01','2029-08-23')]:
  dates=factor.loc[lo:hi].index; z=[]
  for dt in dates:
   ok=factor.loc[dt].notna()&fwd.loc[dt].notna()
   if ok.sum()>=8:z.append(spearmanr(factor.loc[dt][ok],fwd.loc[dt][ok]).statistic)
  if z: print('REG',h,lo,len(z),round(float(np.mean(z)),5),round(float(np.mean(z)/(np.std(z,ddof=1)+1e-12)*np.sqrt(len(z))),5))
 print('SUMMARY',h,'dates',len(arr),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round(np.mean(arr>0),4),'coverage',round(np.mean(cov),4),'turnover',round(float(turnovers),4))
print('instruments',len(ASSETS),'rows',len(prices),'valid_until',prices.index.max())
