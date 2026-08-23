import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).sort_values('date').set_index('date')
 px[s]=d.close.replace(0,np.nan)
prices=pd.DataFrame(px).sort_index().loc[:'2030-11-27']; ret=prices.pct_change()
# Long-window downside-risk normalized reversal: trailing 20d return divided by 40d downside deviation.
down=ret.where(ret<0).rolling(40,min_periods=15).std()
fac=-(prices/prices.shift(20)-1)/(down*np.sqrt(20)); fac=fac.replace([np.inf,-np.inf],np.nan).clip(-8,8)
print('cutoff',prices.index.max().date(),'dates',len(prices),'instruments',len(U))
for h in [5,10,20]:
 fwd=prices.shift(-h)/prices-1; ics=[]; ns=[]; ds=[]
 for dt in prices.index:
  x,y=fac.loc[dt],fwd.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:
   v=spearmanr(x[ok],y[ok]).statistic
   if np.isfinite(v): ics.append(v); ns.append(int(ok.sum())); ds.append(dt)
 a=np.array(ics); print('horizon',h,'valid_dates',len(a),'avg_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,5),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(252),5),'hit',round(np.mean(a>0),5))
 for yr in [2026,2027,2028,2029,2030]:
  z=a[[d.year==yr for d in ds]]
  if len(z): print('regime',yr,len(z),round(z.mean(),8))
print('turnover_proxy',round(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),8))
