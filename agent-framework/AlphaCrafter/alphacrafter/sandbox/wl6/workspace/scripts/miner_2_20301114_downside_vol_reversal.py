import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).sort_values('date').set_index('date')
 px[s]=d.close.replace(0,np.nan)
prices=pd.DataFrame(px).sort_index().loc[:'2030-11-13']
ret=prices.pct_change()
# Downside-risk normalization: penalize only negative daily observations, robust to crypto upside volatility.
down=ret.where(ret<0).rolling(20,min_periods=10).std()
fac=-(prices/prices.shift(20)-1)/(down*np.sqrt(20)); fac=fac.replace([np.inf,-np.inf],np.nan).clip(-8,8)
print('dates',prices.index.min(),prices.index.max(),'instruments',len(U))
for h in [5,10,20]:
 fwd=prices.shift(-h)/prices-1; ics=[]; nobs=[]; dates=[]
 for dt in prices.index:
  x=fac.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:
   v=spearmanr(x[ok],y[ok]).statistic
   if np.isfinite(v): ics.append(v); nobs.append(int(ok.sum())); dates.append(dt)
 a=np.array(ics); print('horizon',h,'valid_dates',len(a),'avg_n',round(np.mean(nobs),3),'coverage',round(np.mean(nobs)/15,5),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(252),5),'hit',round(np.mean(a>0),5))
 for yr in [2026,2027,2028,2029,2030]:
  z=a[[d.year==yr for d in dates]]
  if len(z): print('regime',yr,len(z),round(z.mean(),8))
print('turnover_proxy',round(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),8))
