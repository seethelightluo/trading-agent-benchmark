import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index() for s in A}
p=pd.concat(px,axis=1).sort_index().loc[:'2033-04-13']; r=p.pct_change(); v=r.rolling(20).std(); ret5=p.pct_change(5)
# Reversal is strongest when cross-asset dispersion is elevated; normalize by own risk.
disp=v.mean(axis=1); threshold=disp.rolling(120,min_periods=60).median(); gate=(disp>threshold).astype(float)
sig=(-ret5/v.replace(0,np.nan)).mul(gate,axis=0)
print('dispersion gated reversal; instruments',len(A),'last',p.index[-1].date())
for h in [5,10,20,40]:
 f=p.shift(-h).div(p)-1; vals=[]; ns=[]; dates=[]
 for i in range(len(p)-h):
  z=pd.concat([sig.iloc[i],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(z));dates.append(p.index[i])
 x=np.asarray(vals); ser=pd.Series(x,index=pd.DatetimeIndex(dates))
 print('horizon',h,'dates',len(x),'avg_n',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4),'coverage',round(np.mean(np.asarray(ns)/15),4),'annual',ser.groupby(ser.index.year).mean().round(5).to_dict())
