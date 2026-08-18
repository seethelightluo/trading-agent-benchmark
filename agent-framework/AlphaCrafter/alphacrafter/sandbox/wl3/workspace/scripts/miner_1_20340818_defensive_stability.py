import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 d=get_stock_daily_data(s,days=6000)
 if d is None or len(d)<300:d=get_index_daily_data(s,days=6000)
 return d[['date','close']].copy() if d is not None else None
D={s:fetch(s) for s in U};D={s:d for s,d in D.items() if d is not None}
p=pd.concat({s:d.set_index('date').close for s,d in D.items()},axis=1).sort_index().ffill(); r=np.log(p).diff()
# Defensive stability: inverse realized volatility, penalized by recent drawdown; all lagged.
vol=r.rolling(30).std()*np.sqrt(252)
peak=p.rolling(60).max(); dd=p/peak-1
# low volatility with recovery-aware drawdown penalty, cross-sectionally interpretable
f=(1/(vol+1e-8) * (1+dd.rolling(10).mean())).shift(1)
fr=p.shift(-10)/p-1
rows=[]
for dt in p.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(ic):rows.append((dt,ic,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); mean=q.ic.mean(); sd=q.ic.std(ddof=1)
print('dates',len(q),'avg_n',q.n.mean(),'coverage',len(q)/len(p.index),'IC10',mean,'ICIR10',mean/sd*np.sqrt(252/10),'hit',(q.ic>0).mean(),'turn',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756,1260]:
 z=q.tail(n);print('recent',n,z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1)*np.sqrt(252/10))
print('decay')
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1;a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(h,np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1)*np.sqrt(252/h))
f.to_csv('scripts/miner_1_20340818_defensive_stability_signal.csv',index_label='date');q.to_csv('scripts/miner_1_20340818_defensive_stability_ic.csv')
