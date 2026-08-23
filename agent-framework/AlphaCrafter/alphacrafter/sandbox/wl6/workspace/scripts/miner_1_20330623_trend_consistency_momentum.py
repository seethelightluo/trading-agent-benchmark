import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.concat({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index().loc[:'2033-06-22']; r=p.pct_change(); mom=p.pct_change(20); consistency=(r.gt(0).rolling(20).mean()*2-1); f=mom*consistency
print('candidate trend_consistency_momentum dates',len(p),'instruments',len(U),'last',p.index[-1].date())
for h in [5,10,20,40]:
 fw=p.shift(-h).div(p)-1; vals=[];ns=[];ds=[]
 for dt in p.index[:-h]:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):vals.append(q);ns.append(len(z));ds.append(dt)
 x=np.array(vals); ser=pd.Series(x,index=ds)
 print('horizon',h,'dates',len(x),'avg_n',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4),'coverage',round(np.mean(np.array(ns)/15),4))
 print('annual',ser.groupby(ser.index.year).mean().round(5).to_dict())
print('turnover',round(float(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()),6))
