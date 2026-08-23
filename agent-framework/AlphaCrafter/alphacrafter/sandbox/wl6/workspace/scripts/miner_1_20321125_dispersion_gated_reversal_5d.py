import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].astype(float).sort_index() for s in A}
p=pd.concat(px,axis=1).sort_index().loc[:'2032-11-24']; ret5=p.pct_change(5); disp=ret5.std(axis=1)
gate=(disp>disp.rolling(120,min_periods=60).quantile(.65)); sig=ret5.mul(gate.astype(float),axis=0).mul(-1)
def calc(h):
 f=p.shift(-h).div(p)-1; vals=[]; ns=[]; dates=[]
 for i in range(len(p)-h):
  z=pd.concat([sig.iloc[i].rename('x'),f.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>1:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(z));dates.append(p.index[i])
 x=np.asarray(vals); ser=pd.Series(x,index=pd.DatetimeIndex(dates))
 return len(x),np.mean(ns),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.mean(np.asarray(ns)/15),ser
print('universe',len(A),'data_dates',len(p),'gated_date_rate',round(float(gate.mean()),4))
for h in [5,10,20,40]:
 n,av,ic,ir,hit,cov,ser=calc(h); print('horizon',h,'dates',n,'avg_n',round(av,2),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round(hit,4),'coverage',round(cov,4)); print('regimes',ser.groupby(ser.index.year).mean().round(5).to_dict())
print('max_abs_library_correlation','not_computed')
