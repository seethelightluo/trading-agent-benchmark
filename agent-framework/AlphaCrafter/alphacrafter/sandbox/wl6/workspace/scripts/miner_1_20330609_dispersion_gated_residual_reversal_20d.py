import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index() for s in A}
p=pd.concat(px,axis=1).sort_index().loc[:'2033-06-08']; r=p.pct_change(); ret20=p.pct_change(20)
residual=ret20.sub(ret20.mean(axis=1),axis=0); vol20=r.rolling(20).std(); dispersion=vol20.median(axis=1)
z=dispersion/dispersion.rolling(120,min_periods=60).median(); sig=(-residual).mul(z.clip(0.5,2.0),axis=0)
arr=np.nan_to_num(sig.values,nan=0.0); turnover=np.mean(np.abs(np.diff(np.sign(arr),axis=0)),axis=1).mean()/2
print('idea=dispersion-scaled relative residual reversal instruments',len(A),'last',p.index[-1].date(),'turnover',round(turnover,4))
for h in [5,10,20,40]:
 f=p.shift(-h).div(p)-1; vals=[]; ns=[]; dates=[]
 for i in range(len(p)-h):
  q=pd.concat([sig.iloc[i],f.iloc[i]],axis=1).dropna()
  if len(q)>=8:
   ic=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(ic): vals.append(ic);ns.append(len(q));dates.append(p.index[i])
 x=np.asarray(vals); ser=pd.Series(x,index=pd.DatetimeIndex(dates)); daily=x.mean(); ir=daily/x.std(ddof=1)
 print('horizon',h,'dates',len(x),'avg_n',round(np.mean(ns),2),'IC',round(daily,6),'ICIR',round(ir,6),'hit',round((x>0).mean(),4),'coverage',round(np.mean(np.asarray(ns)/15),4),'annual',ser.groupby(ser.index.year).mean().round(5).to_dict())
