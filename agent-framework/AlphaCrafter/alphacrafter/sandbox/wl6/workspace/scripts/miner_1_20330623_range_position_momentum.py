import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}
p=pd.concat(px,axis=1).sort_index().loc[:'2033-06-22']; r=p.pct_change()
# Continuation signal: 20d momentum, strengthened when price is near the upper/lower
# portion of its 60d range; volatility normalization stabilizes cross-asset scale.
vol=r.rolling(20,min_periods=15).std(); mom=p.pct_change(20); lo=p.rolling(60,min_periods=45).min(); hi=p.rolling(60,min_periods=45).max()
pos=((p-lo)/(hi-lo)).clip(0,1); f=(mom/vol)*(2*pos-1)
print('candidate range_position_momentum; dates',len(p),'instruments',len(U),'last',p.index[-1].date())
def run(h,sub=p.index):
 fw=p.shift(-h).div(p)-1; vals=[]; ns=[]; ds=[]
 for dt in sub[:-h]:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(z));ds.append(dt)
 x=np.array(vals); ser=pd.Series(x,index=ds)
 print('horizon',h,'dates',len(x),'avg_n',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4),'coverage',round(np.mean(np.array(ns)/15),4))
 print('annual',ser.groupby(ser.index.year).mean().round(5).to_dict())
for h in [5,10,20,40]: run(h)
# rank movement turnover
q=f.rank(axis=1,pct=True); print('turnover',round(float(q.diff().abs().mean(axis=1).mean()),6))
