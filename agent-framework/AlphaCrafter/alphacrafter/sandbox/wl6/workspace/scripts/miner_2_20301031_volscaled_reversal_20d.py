import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).sort_values('date').set_index('date')
 px[s]=d.close.replace(0,np.nan)
prices=pd.DataFrame(px).sort_index(); prices=prices.loc[:'2030-10-30']
ret=prices.pct_change(); fac=-(prices/prices.shift(20)-1)/(ret.rolling(20).std()*np.sqrt(20)); fac=fac.clip(-8,8)
for h in [5,10,20]:
 fwd=prices.shift(-h)/prices-1; ics=[]; nobs=[]; dates=[]
 for dt in prices.index:
  x=fac.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8: ics.append(spearmanr(x[ok],y[ok]).statistic); nobs.append(int(ok.sum())); dates.append(dt)
 a=np.array(ics); print(h,len(a),np.mean(nobs),np.mean(nobs)/15,a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),np.mean(a>0))
 for yr in [2026,2027,2028,2029,2030]:
  z=a[[d.year==yr for d in dates]]
  if len(z): print('regime',yr,len(z),z.mean())
print('turnover_proxy',fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
print('dates',prices.index.min(),prices.index.max(),'instruments',len(U))
