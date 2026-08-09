import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index(); r=d.pct_change()
# single idea: inverse realized volatility, lagged; stable defensive cross-sectional preference
f=-r.rolling(20,min_periods=15).std().shift(1)
for h in [1,5,10,20]:
 y=d.shift(-h)/d-1; z=[]; n=[]
 for dt in d.index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:z.append(spearmanr(f.loc[dt,ok],y.loc[dt,ok]).statistic);n.append(ok.sum())
 z=np.array(z);print('H',h,'dates',len(z),'meanN',round(np.mean(n),2),'IC',round(np.mean(z),6),'ICIR',round(np.mean(z)/(np.std(z,ddof=1)+1e-12),6),'hit',round(np.mean(z>0),4),'coverage',round(np.mean(n)/15,4))
print('turnover10',round(f.rank(pct=True).diff(10).abs().mean(axis=1).mean(),6))
for lo,hi in [('2020','2023-12-31'),('2024','2027-12-31'),('2028','2030-12-31'),('2031','2032-08-19')]:
 z=[]
 for dt in d.index:
  if str(dt)[:4]>=lo and str(dt)[:10]<=hi:
   ok=f.loc[dt].notna()&y.loc[dt].notna()
   if ok.sum()>=8:z.append(spearmanr(f.loc[dt,ok],y.loc[dt,ok]).statistic)
 z=np.array(z);print('regime',lo,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/(z.std(ddof=1)+1e-12),6) if len(z)>1 else None)
