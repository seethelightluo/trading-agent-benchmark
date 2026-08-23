import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2030-02-21'); base='../persistent/stock_data'
close=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:cutoff]
# factor: 20d return divided by trailing downside deviation (30 observations), lagged naturally
ret=close.pct_change(); mom=close/close.shift(20)-1
down=ret.where(ret<0).rolling(30,min_periods=3).std()*np.sqrt(30); fac=mom/(down+1e-9)
for h in [1,5,10,20]:
 fwd=close.shift(-h)/close-1; vals=[]; ns=[]; dates=[]
 for d in close.index:
  a=fac.loc[d]; b=fwd.loc[d]; ok=a.notna()&b.notna()
  if ok.sum()>=8:
   vals.append(spearmanr(a[ok],b[ok]).statistic); ns.append(ok.sum()); dates.append(d)
 x=np.asarray(vals); print('horizon',h,'dates',len(x),'avg_n',np.mean(ns),'coverage',np.sum(ns)/(len(ns)*15),'IC',x.mean(),'ICIR',x.mean()/(x.std(ddof=1)+1e-12)*np.sqrt(len(x)),'hit',(x>0).mean())
 yy=pd.DataFrame({'ic':x,'date':dates}); print(yy.assign(year=pd.to_datetime(yy.date).dt.year).groupby('year').ic.mean().round(4).to_dict())
