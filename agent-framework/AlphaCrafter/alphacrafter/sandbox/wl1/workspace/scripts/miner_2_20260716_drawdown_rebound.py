import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index(); r=p.pct_change()
# Drawdown rebound: distance below trailing 60d high, an interpretable medium-horizon mean-reversion signal.
f=1-p/p.rolling(60,min_periods=40).max()
for h in [1,5,10]:
 ic=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],r.shift(-h).loc[d]],axis=1).dropna()
  if len(z)>=8: ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.array(ic); print('h',h,'dates',len(a),'avg_n',np.mean(ns),'IC',a.mean(),'absIC',abs(a.mean()),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'coverage',np.mean(ns)/15)
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
